import os
import streamlit as st
import openai
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from qdrant_client import qdrant_client


env_path = Path(__file__).parent.parent / ".env"
success = load_dotenv(dotenv_path=env_path, override=True)
openai.api_key = os.environ["OPENAI_API_KEY"]
qdrant_key = os.environ["QDRANT_API_KEY"]
qdrant_host = os.environ["QDRANT_HOST"]

@st.cache_resource
def load_chain():

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()
    
    # Load OpenAI chat model
    llm = ChatOpenAI(
            model="gpt-4-0314",
            temperature=0,
            )
    
    # Prepare vector store
    client = qdrant_client.QdrantClient(
        qdrant_host,
        api_key=qdrant_key,
        )

    vector_store = Qdrant(
        client=client, 
        collection_name="raven", 
        embeddings=embeddings,
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history", output_key="answer")
        
    # Create system prompt
    template = """
    You are an AI assistant for answering questions about laboratory regulatory matters.
    You are given the following extracted text from a list of regulations and a question. 
    Provide a professional and complete answer.
    Base your answer solely on the information provided in the prompts.
    Do not make up answers or provide answers from sources other than the extracted text.
    Provide a reference for each assertion you make.
    If you don't know the answer, say 'Sorry, I am not sure.'. 
    If the question is not about laboratory policies or regulations, inform them that you are tuned to only answer questions about laboratory regulations.

    {context}
    Question: {question}
    Helpful Answer:"""
        
    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(  llm=llm, 
                                                    retriever=retriever, 
                                                    memory=memory, 
                                                    get_chat_history=lambda h : h,
                                                    verbose=True,
                                                    return_source_documents=True,
                                                    )
    
    # Add system prompt to chain
    chain_prompt = PromptTemplate(input_variables=["context", "question"],template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=chain_prompt)
    
    return chain





