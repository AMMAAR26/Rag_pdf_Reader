## 🤖 RAG PDF Q&A Chatbot


## 📌 Overview

**RAG PDF Q&A** is a smart, minimal app where you can upload any PDF and start asking questions about it. It uses Retrieval-Augmented Generation (RAG) powered by:

- 🧠 **LangChain** for intelligent document retrieval
- ⚙️ **Transformers** for LLM-driven answers
- 🧰 **ChromaDB** for vector storage
- 🎯 **Streamlit** for an interactive web interface

---

## 🚀 Features

- 📂 Upload PDFs up to 200MB
- 🤖 Ask natural language questions about the document
- 💬 Simple and interactive chatbot UI
- 🔍 RAG-based context-aware answers with citations
- 💾 Local vector DB support via Chroma




## 🛠️ Project Structure

├── .gitattributes                 # Git settings
├── README.md                      # This file
├── app.py                         # Streamlit frontend
├── rag_backend.py                 # All backend functions (split, embed, chain)
├── rag-for-pdf-reader.ipynb       # Jupyter notebook version
├── uploaded.pdf                   # uploaded PDF (used by app)
├── chroma_db/                     # Local vector DB folder
├── __pycache__/                   # Python cache folder



## ⚙️ How It Works
Upload a PDF via Streamlit UI

The file is split into smaller chunks

Embeddings are generated using all-mpnet-base-v2

Chunks are stored in ChromaDB

RAG retrieves the most relevant parts

LLM generates an answer based on context


## 💻 Run the App ▶️ Local Environment


** Clone the repo**
git clone https://github.com/Ammaar26/rag-pdf-chatbot.git
cd rag-pdf-chatbot

** Install dependencies**
pip install -r requirements.txt

**# Run the Streamlit app**
streamlit run app.py 

## 📬 Authors

- **Mohammad Sir** – [@mohammad-Ammar](https://github.com/AMMAAR26)  
- **Maverick** – [@maverick0810](https://github.com/maverick0810)


## 📬 Contact

 mohdammar026@gmail.com 
📫 Feel free to open an issue or PR if you'd like to contribute, suggest features, or report bugs.




