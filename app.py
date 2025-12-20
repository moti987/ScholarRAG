import os
import streamlit as st
from textwrap import shorten
from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(page_title="ScholarRAG", page_icon="📚")
st.title("📚 ScholarRAG")
st.markdown("Ask questions from your **research papers** using Retrieval-Augmented Generation.")

# -------------------------------------------------
# GROQ API KEY (NO HARDCODING)
# -------------------------------------------------
st.sidebar.header("Configuration")
groq_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()

client = Groq(api_key=groq_key)

# -------------------------------------------------
# LOAD VECTOR DATABASE
# -------------------------------------------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local("vectorstore/db_faiss", embeddings)

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 4})

st.sidebar.success("Knowledge base loaded")

# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# CHAT INPUT
# -------------------------------------------------
if user_input := st.chat_input("Ask a question from your papers…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and generating answer..."):
            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
You are a research assistant.
Answer ONLY using the context below.
If the answer is not present, say you could not find it in the papers.

Context:
{context}

Question:
{user_input}

Answer:
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            answer = response.choices[0].message.content
            st.markdown(answer)

            st.markdown("### 📄 Sources")
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown").split("/")[-1]
                snippet = shorten(d.page_content.replace("\n", " "), width=180)
                st.markdown(f"**{i}. {src}**")
                st.caption(snippet)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
