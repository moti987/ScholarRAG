
# 📘 ScholarRAG  
**An Evidence-Grounded Research Assistant using Retrieval-Augmented Generation**

---

## 🧠 Overview

**ScholarRAG** is an AI-powered research assistant designed to answer questions strictly grounded in academic documents.  
It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant passages from research papers and generate answers only from those sources.

This reduces hallucinations and ensures transparency, making the system suitable for academic use.

---

## 🎯 Key Objectives

- Provide accurate, citation-grounded answers  
- Reduce hallucinations in LLM responses  
- Maintain academic integrity  
- Enable conversational research exploration  

---

## 🏗️ System Architecture

ScholarRAG follows a retrieval-augmented generation pipeline.

```

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

```

---

## 🛠️ Technology Stack

| Component | Technology |
|---------|------------|
| Frontend | Streamlit |
| LLM | Groq (LLaMA-3.1) |
| Vector Database | FAISS |
| Embeddings | Sentence-Transformers (MiniLM) |
| Framework | LangChain |
| Language | Python |

---

## ✨ Features

- Chat-style interface with preserved history  
- Evidence-grounded answers  
- Citation chips (paper · page number)  
- No-answer handling for out-of-scope queries  
- Light and dark UI themes  
- Clear chat functionality  

---

## 📂 Project Structure

```

ScholarRAG/
├── app.py
├── create_knowledge_base.py
├── cli_app.py
├── nlp_research_corpus/
├── vectorstore/
├── metadata.csv
├── metadata.json
└── README.md

````

---

## 🚀 How to Run

```bash
python -m venv BDAproj
BDAproj\Scripts\activate
pip install streamlit groq langchain langchain-community faiss-cpu sentence-transformers
python create_knowledge_base.py
streamlit run app.py
````

---

## 🧪 Testing

* Functional queries
* Multi-source questions
* Rephrased and ambiguous queries
* Out-of-scope questions to prevent hallucination
* UI and usability testing

---

## 🎓 Academic Note

ScholarRAG prioritises academic integrity.
If relevant information is not found in the document corpus, the system explicitly states this instead of generating speculative answers.

---

ScholarRAG – Retrieval-Augmented Research Assistant


