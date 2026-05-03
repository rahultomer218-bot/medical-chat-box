import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def get_qa_chain(db):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.5,
        max_tokens=512
    )
    prompt = PromptTemplate(
        template="""You are a helpful medical assistant. Use the context below to answer the question as helpfully as possible.
If the context does not contain enough information, use your general medical knowledge but clearly mention it is general advice.
Always recommend consulting a doctor for personal medical concerns.

Context: {context}
Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def main():
    st.title("🏥 Medical Chatbot")
    st.caption("Ask me anything about medical topics from the loaded books.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    db = get_vectorstore()

    prompt = st.chat_input("Ask your medical question here...")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.spinner("Thinking..."):
            qa_chain = get_qa_chain(db)
            result = qa_chain.invoke({"query": prompt})
            response = result["result"]

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()