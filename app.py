import streamlit as st
from retriever import DocumentRetriever
from rag_chain import run_rag_pipeline
import os
import tempfile
import pandas as pd

st.set_page_config(page_title="AskAnyDoc", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>📄 AskAnyDoc — Ask Anything from Your Files</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload any document — CSV, TXT, PDF, or DOCX — and get intelligent answers instantly.</p>", unsafe_allow_html=True)

if 'retriever' not in st.session_state:
    st.session_state.retriever = DocumentRetriever()
    st.session_state.index = None
    st.session_state.texts = []

uploaded_file = st.file_uploader("Upload your document here", type=["csv", "txt"], label_visibility="collapsed")

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    if file_ext == "csv":
        df = pd.read_csv(tmp_path)
        text_chunks = df.apply(lambda row: ' '.join(map(str, row.dropna())), axis=1).tolist()
    elif file_ext == "txt":
        with open(tmp_path, 'r', encoding='utf-8') as f:
            text_chunks = f.read().split("\n\n")
    else:
        st.error("Unsupported file type.")
        st.stop()

    with st.spinner("Indexing your document..."):
        embeddings = st.session_state.retriever.model.encode(text_chunks, show_progress_bar=True)
        index, texts = st.session_state.retriever.build_faiss_index(embeddings, text_chunks)
        st.session_state.index = index
        st.session_state.texts = texts
        st.success("🧠 Document indexed. Ask away!")

if st.session_state.index is not None:
    with st.form("qna"):
        query = st.text_input("Ask a question from your document", placeholder="e.g. What are the key findings?")
        submitted = st.form_submit_button("Get Answer")

        if submitted and query:
            with st.spinner("Thinking..."):
                answer = run_rag_pipeline(
                    query,
                    st.session_state.retriever,
                    st.session_state.index,
                    st.session_state.texts
                )
                st.markdown("### 🧠 Answer:")
                st.success(answer)
