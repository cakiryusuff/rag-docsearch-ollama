# 🧠 RAG-Based PDF Q&A with FAISS / Chroma + Ollama

This project implements a simple Retrieval-Augmented Generation (RAG) system to answer questions using local PDF files. It uses semantic search (via FAISS or Chroma) to retrieve relevant document chunks and then passes them to a locally hosted LLM using [Ollama](https://ollama.com) (e.g., `qwen2.5`) for answer generation.

---

## 🚀 Features

- 📄 Loads PDF files from a local directory
- ✂️ Splits documents into chunks using `RecursiveCharacterTextSplitter`
- 🔍 Performs vector-based semantic search with **FAISS** or **Chroma**
- 🤖 Sends retrieved context to a local **Ollama LLM**
- 💬 Interactive CLI interface for asking questions

---

## 🛠️ Installation

### (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

Note: For FAISS, use faiss-cpu (works on most systems):
```bash
pip install faiss-cpu
```

## ⚙️ Usage
```bash
python main.py
```
You will be prompted to:

1. Enter your question

2. Choose a vector store (FAISS or Chroma)

The top relevant chunks will be retrieved and passed to an LLM for answer generation.

Note: Make sure Ollama is installed and running. On your local machine.
