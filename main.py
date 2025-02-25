import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv() # takes environment variables from .env


# Initialise LLM with required params
llm = ChatGroq(
    model = "mixtral-8x7b-32768",
    temperature= 0.9,
    max_tokens= 500
)

st.title("Articles QA Bot")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path ="faiss_store_hf.pkl"

main_placefolder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls = urls)
    main_placefolder.text("Data Loading.... Started... ")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size = 1000
    )
    main_placefolder.text("Text Splitting.... Started... ")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS Index
    embeddings = HuggingFaceEmbeddings()
    vectorstore_hf= FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Embedding Vector Started Building.... ")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path,"wb") as f:
        pickle.dump(vectorstore_hf, f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm =llm, retriever= vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)