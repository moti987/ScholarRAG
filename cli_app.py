import os
import sys
from textwrap import shorten

print("------------------------------------------------------------------")
print("1. ⏳ Starting up...")
print("2. 📦 Importing libraries...")

from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("3. ✅ Libraries Imported Successfully!")

# ------------------------------------------------------------
# 1) GROQ API KEY
# ------------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not api_key:
    api_key = input("\n🔑 Enter your GROQ API Key (Right Click to Paste): ").strip()

if not api_key:
    print("❌ No Groq key provided. Exiting.")
    sys.exit(1)

client = Groq(api_key=api_key)

# ------------------------------------------------------------
# 2) LOAD FAISS KNOWLEDGE BASE
# ------------------------------------------------------------
print("\n4. 🧠 Loading the Vector Database...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    db = FAISS.load_local("vectorstore/db_faiss", embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    print("   ✅ Knowledge Base Loaded!")
except Exception as e:
    print(f"   ❌ Error loading database: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# 3) CHAT LOOP (PURE RAG)
# ------------------------------------------------------------
print("\n" + "=" * 55)
print("🚀 SYSTEM READY! (Groq + FAISS RAG)")
print("Type 'exit' to quit")
print("=" * 55 + "\n")

while True:
    query = input("You: ").strip()
    if query.lower() in {"exit", "quit"}:
        break
    if not query:
        continue

    try:
        print("\n🤖 Retrieving context...\n")

        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are a research assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say:
"I could not find this information in the provided papers."

Context:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content
        print("Answer:\n", answer, "\n")

        print("Sources used:")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            snippet = shorten(d.page_content.replace("\n", " "), width=160, placeholder="...")
            print(f" {i}. {src}")
            print(f"    ↳ {snippet}")

        print("\n" + "-" * 55 + "\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")
