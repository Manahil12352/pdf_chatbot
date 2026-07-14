import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")

@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading and processing PDF..."):
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    st.success(f"PDF processed! {len(chunks)} chunks created. Ask away.")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):
            # Step 1: retrieve relevant chunks
            relevant_docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Step 2: build the prompt manually
            prompt = f"""Answer the question based only on the following context. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

            # Step 3: call the LLM directly
            response = llm.invoke(prompt)

            st.write("### Answer")
            st.write(response.content)

            with st.expander("Show retrieved context"):
                st.write(context)

    os.remove(temp_path)
else:
    st.info("Upload a PDF above to get started.")
