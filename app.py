import streamlit as st
from textwrap import shorten
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="ScholarRAG",
    layout="wide"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("⚙️ Settings")

theme_mode = st.sidebar.radio(
    "Theme",
    ["Light (Muted Green)", "Dark (Muted Green)"],
    index=0
)

groq_key = st.sidebar.text_input("Groq API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **ScholarRAG**

    📖 Evidence-grounded research assistant  
    🧠 Retrieval-Augmented Generation  
    🗃️ FAISS + Groq (LLaMA-3.1)  

    Designed for academic research.
    """
)

if not groq_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()

client = Groq(api_key=groq_key)

# -------------------------------------------------
# THEME STYLES (NO BACKGROUND BARS)
# -------------------------------------------------
if theme_mode == "Light (Muted Green)":
    css = """
    <style>
    .stApp { background-color: #FAFBFA; color: #1F3F36; }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 { color: #2F6F5E; font-weight: 700; text-align: center; }
    h2, h3 { color: #4F8F7B; font-weight: 600; }

    .card {
        background: #FFFFFF;
        border: 1px solid #E3E8E6;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 26px;
    }

    .answer-box {
        background-color: #E6F2EE;
        border-left: 6px solid #2F6F5E;
        padding: 18px;
        border-radius: 10px;
    }

    .chip {
        display: inline-block;
        background-color: #F0F7F4;
        border: 1px solid #CFE6DE;
        color: #2F6F5E;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 13px;
        margin: 4px 6px 4px 0;
    }

    input {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CFE6DE !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }

    hr { background-color: #E3E8E6; height: 1px; border: none; }
    </style>
    """
else:
    css = """
    <style>
    .stApp { background-color: #0F1E1A; color: #E6F2EE; }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 { color: #7FC7AE; font-weight: 700; text-align: center; }
    h2, h3 { color: #9FD9C5; font-weight: 600; }

    .card {
        background-color: #162E27;
        border: 1px solid #234A3E;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 26px;
    }

    .answer-box {
        background-color: #162E27;
        border-left: 6px solid #7FC7AE;
        padding: 18px;
        border-radius: 10px;
    }

    .chip {
        display: inline-block;
        background-color: #1F3D34;
        border: 1px solid #234A3E;
        color: #9FD9C5;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 13px;
        margin: 4px 6px 4px 0;
    }

    input {
        background-color: #162E27 !important;
        border: 1.5px solid #234A3E !important;
        border-radius: 8px !important;
        padding: 12px !important;
        color: #E6F2EE !important;
    }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1>📚 ScholarRAG</h1>
    <p style="text-align:center; font-size:17px;">
        An Evidence-Grounded AI Assistant for Academic Research
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD VECTOR DB
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

# -------------------------------------------------
# QUESTION CARD
# -------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🧠 Ask a Research Question")

query = st.text_input(
    "Enter a question grounded in your uploaded research papers",
    placeholder="e.g. What is retrieval-augmented generation?"
)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# PROCESS QUERY
# -------------------------------------------------
if query:
    with st.spinner("🔍 Retrieving evidence and generating answer..."):
        docs = retriever.get_relevant_documents(query)

        if not docs:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### ⚠️ No Answer Found")
            st.info(
                "I could not find relevant information in the uploaded papers. "
                "Try rephrasing your question or asking about a different topic."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are a research assistant.
Answer ONLY using the context below.
If the answer is not present, say you could not find it in the papers.

Context:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()

    # -------------------------------------------------
    # ANSWER
    # -------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ✨ Answer")

    if "could not find" in answer.lower():
        st.warning("The system did not find a grounded answer in the papers.")
    else:
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # SOURCES WITH CITATION CHIPS
    # -------------------------------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🗃️ Retrieved Evidence")

    for d in docs:
        src = d.metadata.get("source", "unknown").split("/")[-1]
        page = d.metadata.get("page", "–")
        st.markdown(
            f"<span class='chip'>{src} · Page {page}</span>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
