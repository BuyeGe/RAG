# 🚀 Local RAG Information Retrieval System

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)
![ChromaDB](https://img.shields.io/badge/Chroma-Vector_DB-orange.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow.svg)

## 📌 Project Overview
This project is an end-to-end **Information Retrieval System** built from scratch, serving as the core foundation for a Retrieval-Augmented Generation (RAG) pipeline. It autonomously crawls web data, processes text into vector embeddings using local open-source models, and performs high-speed semantic search.

This project was built entirely using local, free models without relying on paid external APIs, ensuring maximum data privacy and cost-efficiency.

## ✨ Key Features
- **Automated Web Crawler:** Uses `BeautifulSoup4` and LangChain's `WebBaseLoader` to extract clean, readable text from target URLs.
- **Smart Text Chunking:** Implements `RecursiveCharacterTextSplitter` with intelligent chunk overlap to maintain semantic context boundaries.
- **Local Vector Embeddings:** Utilizes HuggingFace's lightweight `all-MiniLM-L6-v2` model to transform text into numerical vectors entirely on the local machine.
- **Vector Database (ChromaDB):** Persists embedded data locally for fast, reliable, and scalable semantic similarity search.

## 🛠️ Tech Stack
- **Core Language:** Python 3.12
- **AI Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embedding Model:** HuggingFace `sentence-transformers`
- **Data Ingestion:** BeautifulSoup4

## 🚀 How to Run (Quick Start)

**1. Clone the repository & Install dependencies:**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt