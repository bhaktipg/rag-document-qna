import os
import streamlit as st
from loguru import logger
from app.rag.ingestion import load_document, chunk_document
from app.rag.chain import RAGChain
from app.rag.vectorstore import list_collections, delete_collection, add_documents, _sanitize_name

def render_sidebar(rag_chain: RAGChain):
    with st.sidebar:
        st.title("📁 Document Control")
        
        # 1. File Upload Area
        st.subheader("Ingest New Documents")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # We save temporary files to ingest them
            temp_dir = "./data/temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            for file in uploaded_files:
                # Avoid duplicate ingestion per session if possible
                col_name = _sanitize_name(file.name)
                existing_collections = list_collections()
                
                if col_name in existing_collections:
                    continue
                    
                temp_path = os.path.join(temp_dir, file.name)
                try:
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                        
                    with st.spinner(f"Ingesting {file.name}..."):
                        docs = load_document(temp_path)
                        if docs:
                            chunks = chunk_document(docs)
                            add_documents(chunks)
                            st.success(f"Successfully loaded {file.name}!")
                        else:
                            st.warning(f"File {file.name} had no extractable text or was too short.")
                except Exception as e:
                    st.error(f"Error ingesting {file.name}: {e}")
                    logger.error(e)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        st.divider()
        
        # 2. List Ingested Collections
        st.subheader("Ingested Documents")
        collections = list_collections()
        if not collections:
            st.info("No documents ingested yet.")
        else:
            # We track selected collections for active search
            st.markdown("Select documents to query:")
            selected_cols = []
            for col in collections:
                # Use human-friendly label
                col_display = col
                col_display = col_display.replace("_", ".")
                
                col1, col2 = st.columns([0.8, 0.2])
                active = col1.checkbox(col_display, value=True, key=f"active_{col}")
                if active:
                    selected_cols.append(col)
                    
                if col2.button("🗑️", key=f"del_{col}"):
                    try:
                        delete_collection(col)
                        st.success(f"Deleted {col_display}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not delete: {e}")
            st.session_state.selected_collections = selected_cols

            
        st.divider()
        
        # 4. LLM settings
        st.subheader("Model Configuration")
        
        default_provider = os.getenv("LLM_PROVIDER", "ollama")
        provider = st.selectbox(
            "LLM Provider",
            ["ollama", "groq"],
            index=0 if default_provider == "ollama" else 1
        )
        st.session_state.llm_provider = provider
        
        if provider == "ollama":
            default_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            model = st.text_input("Ollama Model", value=default_model)
        else:
            default_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            model = st.text_input("Groq Model", value=default_model)
            
        st.session_state.llm_model = model
        
        # Top K Slider
        default_top_k = int(os.getenv("TOP_K_RETRIEVAL", 5))
        top_k = st.slider("Citations (Top K Chunks)", min_value=1, max_value=10, value=default_top_k)
        st.session_state.top_k = top_k
        
        # Optional custom system prompt
        st.divider()
        st.caption("Powered by ChromaDB & SentenceTransformers.")