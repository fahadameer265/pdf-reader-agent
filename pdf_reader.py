import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

st.set_page_config(page_title="PDF Reader Agent", page_icon="📄")
st.title("📄 PDF Reader Agent")
st.write("PDF upload karo aur koi bhi sawaal poochho!")

uploaded_file = st.file_uploader("PDF yahan upload karo", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    with st.spinner("PDF padh raha hoon... ⏳"):
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        embeddings = FastEmbedEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    st.success("✅ PDF ready hai! Ab sawaal poochho!")
    question = st.text_input("Apna sawaal yahan likho:")
    if question:
        with st.spinner("Jawab dhundh raha hoon... 🔍"):
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                temperature=0.3
            )
            prompt = PromptTemplate.from_template("""
Neeche diye gaye context ke basis pe sawaal ka jawab do.
Agar jawab context mein nahi hai toh kaho "Mujhe is baare mein maloom nahi".
Context: {context}
Sawaal: {question}
Jawab:
""")
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            answer = chain.invoke(question)
        st.write("### 🤖 Jawab:")
        st.write(answer)
        os.unlink(tmp_path)