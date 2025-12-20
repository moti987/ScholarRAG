import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# STEP-1: Setup the paths
DATA_PATH = "nlp_research_corpus"
DB_FAISS_PATH = "vectorstore/db_faiss"

#STEP-2: Loading the PDFS
print(f"loading PDF files from {DATA_PATH}...")
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()
print(f"loaded{len(documents)}pages from the PDF.")

#STEP-3: Chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")
#STEP-3: Embeddings
print("Generating embeddings... (This might take a minute)")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

#STEP-5: VECTOR DATABASE:

print("building the vector database..")
db = FAISS.from_documents(texts, embeddings)
#STEP-6 Saving It Locally
db.save_local(DB_FAISS_PATH)
print(f"✅ Success! Vector Database saved to '{DB_FAISS_PATH}'")