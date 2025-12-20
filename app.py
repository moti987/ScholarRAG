import streamlit as st
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
    ["Light", "Dark"],
    index=0
)

groq_key = st.sidebar.text_input("Groq API Key", type="password")

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.history = []


st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **ScholarRAG**

    -> Evidence-grounded research assistant  
    
    -> Retrieval-Augmented Generation  
    
    -> FAISS + Groq (LLaMA-3.1)
    """
)

if not groq_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()

client = Groq(api_key=groq_key)

# -------------------------------------------------
# CLEAN, NEUTRAL STYLES (NO GREEN BOXES)
# -------------------------------------------------
if theme_mode == "Light":
    css = """
    <style>
    .stApp { background-color: #FAFBFA; color: #1F3F36; font-size: 14px; }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 { text-align: center; font-size: 28px; }
    h2, h3 { font-size: 16px; }

    .citation {
        display: inline-block;
        font-size: 12px;
        padding: 2px 8px;
        margin-right: 6px;
        border-radius: 6px;
        border: 1px solid #DADADA;
        color: #444;
    }

    .logo {
    font-size: 26px;
    margin-right: 8px;
    color: #2F6F5E;
    }

    </style>
    """
else:
    css = """
    <style>
    .stApp { background-color: #0F1E1A; color: #E6F2EE; font-size: 14px; }

    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 { text-align: center; font-size: 28px; }

    .citation {
        display: inline-block;
        font-size: 12px;
        padding: 2px 8px;
        margin-right: 6px;
        border-radius: 6px;
        border: 1px solid #355E52;
        color: #9FD9C5;
    }
    .logo {
    font-size: 26px;
    margin-right: 8px;
    color: #7FC7AE;
    }

    </style>
    """

st.markdown(css, unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1>
        <span class="logo">📘</span>ScholarRAG
    </h1>
    <p style='text-align:center; font-size:14px; margin-top:-6px;'>
        Evidence-Grounded AI Assistant for Academic Research
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")


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
# CHAT HISTORY (NEW, MINIMAL)
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Render history
for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])

    with st.chat_message("assistant"):
        st.markdown(item["answer"])

        if item["citations"]:
            st.markdown("**Sources:**")
            for c in item["citations"]:
                st.markdown(f"<span class='citation'>{c}</span>", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT
# -------------------------------------------------
query = st.chat_input("Ask a research question")

# -------------------------------------------------
# PROCESS QUERY (UNCHANGED LOGIC)
# -------------------------------------------------
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.get_relevant_documents(query)

            if not docs:
                answer = "I could not find relevant information in the uploaded papers."
                citations = []
            else:
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
                citations = []

                for d in docs:
                    src = d.metadata.get("source", "unknown").split("/")[-1]
                    page = d.metadata.get("page", "–")
                    citations.append(f"{src} · Page {page}")

            st.markdown(answer)

            if citations:
                st.markdown("**Sources:**")
                for c in citations:
                    st.markdown(f"<span class='citation'>{c}</span>", unsafe_allow_html=True)

    st.session_state.history.append(
        {"question": query, "answer": answer, "citations": citations}
    )
