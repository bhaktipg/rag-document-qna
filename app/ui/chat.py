import streamlit as st
from app.rag.chain import RAGChain

def render_chat_interface(rag_chain: RAGChain):
    st.title("💬 RAG Document Q&A")
    st.caption("Ask questions about your uploaded documents and get answers with exact source context.")
    
    # 1. Render historical chat logs
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Viewed Sources"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"**Source {idx+1} ({src['source_file']}) — Chunk {src['chunk_id']}:**")
                        st.caption(f"\"{src['text']}\"")
                        
    # 2. User Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Append and render user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Call RAG Backend and render Assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Fetch selections from state
            provider = st.session_state.get("llm_provider", "ollama")
            model_name = st.session_state.get("llm_model")
            top_k = st.session_state.get("top_k", 5)
            collections = st.session_state.get("selected_collections", [])
            
            with st.spinner("Retrieving sources & thinking..."):
                try:
                    # Run RAG streaming
                    stream, chunks = rag_chain.ask_stream(
                        query=prompt,
                        provider=provider,
                        model_name=model_name,
                        top_k=top_k,
                        collections=collections
                    )
                    
                    # Consume generator and stream to UI
                    full_response = ""
                    for chunk in stream:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    
                    # Display sources expander below
                    if chunks:
                        with st.expander("📚 Viewed Sources"):
                            for idx, src in enumerate(chunks):
                                st.markdown(f"**Source {idx+1} ({src['source_file']}) — Chunk {src['chunk_id']}:**")
                                st.caption(f"\"{src['text']}\"")
                                
                    # Append Assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": chunks
                    })
                    
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
                    st.error("Please verify that your chosen LLM service is running and API keys are set correctly.")