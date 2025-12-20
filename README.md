📘 ScholarRAG

An Evidence-Grounded Research Assistant using Retrieval-Augmented Generation

🧠 Overview

ScholarRAG is an AI-powered research assistant designed to answer questions strictly grounded in academic documents.
Instead of relying on a language model’s internal knowledge, ScholarRAG uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from research papers and generate answers only from those sources.

This approach reduces hallucinations, improves transparency, and makes the system suitable for academic and research-oriented use cases.

🎯 Key Objectives

Provide accurate, citation-grounded answers from research papers

Reduce hallucinations commonly seen in standard LLM chatbots

Maintain academic integrity by showing evidence sources

Offer a clean, conversational interface for research exploration

🏗️ System Architecture

ScholarRAG follows a retrieval-augmented generation pipeline, where document retrieval and answer generation are tightly coupled.

🔧 Architecture Diagram (Conceptual)
┌──────────────────────┐
│   Research Papers    │
│   (PDF Documents)    │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Document Chunking   │
│  & Preprocessing     │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Text Embeddings     │
│  (Sentence-Transformers)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  FAISS Vector Store  │
│  (Semantic Index)    │
└─────────┬────────────┘
          │
 User Query
          │
          ▼
┌──────────────────────┐
│ Query Embedding      │
│ & Similarity Search  │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Retrieved Context    │
│ (Top-K Chunks)       │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  LLM (Groq – LLaMA)  │
│  Grounded Generation │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Answer + Citations   │
│ (Paper · Page No.)   │
└──────────────────────┘

🧠 Architectural Rationale

FAISS enables fast semantic similarity search over large document collections

Sentence-transformer embeddings capture semantic meaning beyond keywords

Groq-hosted LLaMA models generate fluent answers while respecting retrieved context

Strict prompting ensures the model answers only from retrieved evidence

This architecture ensures that answers are traceable, verifiable, and academically reliable.

🛠️ Technology Stack
Component	Technology
Frontend	Streamlit
LLM	Groq (LLaMA-3.1)
Vector Database	FAISS
Embeddings	Sentence-Transformers (MiniLM)
Framework	LangChain
Language	Python
✨ Features

💬 Chat-style interface with preserved conversation history

📄 Evidence-grounded answers (no unsupported claims)

🏷️ Citation chips showing paper name and page number

⚠️ No-answer detection when information is not found

🌓 Light and dark themes for readability

🧹 Clear chat functionality for repeated testing

📂 Project Structure
ScholarRAG/
├── app.py                     # Streamlit application
├── create_knowledge_base.py   # Builds FAISS vector store
├── cli_app.py                 # CLI-based RAG interface
├── nlp_research_corpus/       # Research PDFs
├── vectorstore/               # FAISS index (generated)
├── metadata.csv / json        # Document metadata
└── README.md

🚀 How to Run the Project
1️⃣ Create and activate a virtual environment
python -m venv BDAproj
BDAproj\Scripts\activate

2️⃣ Install dependencies
pip install streamlit groq langchain langchain-community faiss-cpu sentence-transformers

3️⃣ Build the knowledge base
python create_knowledge_base.py

4️⃣ Run the application
streamlit run app.py


Enter your Groq API key in the sidebar when prompted.

🧪 Testing & Validation

The system was tested using:

Functional queries (definitions, explanations)

Multi-source questions

Rephrased and ambiguous queries

Out-of-scope questions to verify non-hallucination

UI and usability tests

🎓 Academic Note

ScholarRAG is designed with academic integrity in mind.
If relevant information is not found in the documents, the system explicitly states this instead of generating speculative answers.

👤 Author

Hamza
ScholarRAG – Retrieval-Augmented Research Assistant