import streamlit as st
from rag_backend import load_and_split_pdf, create_vectordb, get_llm, get_rag_chain
import os

st.set_page_config(page_title="RAG PDF Q&A 🤖📄", page_icon=":books:", layout="wide")
st.title("RAG PDF Q&A 🤖📄")
st.markdown("Ask questions about your PDF! Upload a file, and chat with your document. ✨")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        st.success("PDF uploaded! 🎉")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "sources" not in st.session_state:
    st.session_state.sources = []

# Load and process PDF
if uploaded_file:
    if st.session_state.rag_chain is None:
        with st.spinner("Processing PDF... ⏳"):
            splits = load_and_split_pdf("uploaded.pdf")
            vectordb = create_vectordb(splits)
            llm = get_llm()
            rag_chain = get_rag_chain(vectordb, llm)
            st.session_state.rag_chain = rag_chain
        st.success("Ready to chat! 🚀")

    # Chat interface
    st.subheader("💬 Chat with your PDF")
    user_input = st.text_input("Type your question here... 🤔", key="user_input")
    if st.button("Ask", use_container_width=True):
        if user_input.strip():
            with st.spinner("Thinking... 🤖"):
                response = st.session_state.rag_chain.invoke({"input": user_input})
                answer = response.get("answer", "No answer found.")
                sources = response.get("context", [])
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", answer))
                st.session_state.sources = sources
                st.balloons()
        else:
            st.warning("Please enter a question! 📝")

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧑‍💻 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")

    # Show sources
    if st.session_state.sources:
        st.markdown("**🔗 Sources:**")
        for i, doc in enumerate(st.session_state.sources):
            st.markdown(f"- Chunk {i+1}: `{doc.page_content[:100]}...`")
else:
    st.info("Upload a PDF to get started! 📄")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Langchain, and Llama 2.0") 