import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.5,
        max_tokens=512
    )

prompt = PromptTemplate(
    template="Use the context to answer. Say you don't know if unsure.\nContext: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

user_query = input("Write your query here: ")
response = qa_chain.invoke({"query": user_query})
print("Answer:", response["result"])