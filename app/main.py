import os
import streamlit as st
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

from app.rag.chain import RAGChain
from app.ui.sidebar import render_sidebar
from app.ui.chat import render_chat_interface

# 1. Page Configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling adjustments (CSS)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2d2d2d;
        border-radius: 4px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Initialize RAG Chain (cached in session state so we don't reload embeddings model on every page rerun)
if "rag_chain" not in st.session_state:
    with st.spinner("Loading embedding model, please wait..."):
        st.session_state.rag_chain = RAGChain()

# 3. Initialize Chat Log History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Drop some documents in the sidebar, make sure they are checked, and ask me anything.", "sources": []}
    ]

# 4. Render Sidebar and Chat Interfaces
render_sidebar(st.session_state.rag_chain)
render_chat_interface(st.session_state.rag_chain)