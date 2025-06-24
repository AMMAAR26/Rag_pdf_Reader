from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

# 1. Load PDF and split

def load_and_split_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
    return splitter.split_documents(docs)

# 2. Embeddings and vector store
def create_vectordb(splits, persist_directory="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    return vectordb

# 3. LLM
def get_llm():
    # model_id = "meta-llama/Llama-2-7b-chat-hf"  # This model is very large (14GB) and requires authentication.
    model_id = "google/flan-t5-base" # Using a smaller, public model for faster testing.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# 4. RAG Chain (new style)
def get_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the provided context:\n<context>\n{context}\n</context>\n\nQuestion: {input}"
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain) 