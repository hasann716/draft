from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import ConfigurableField, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from retry import retry
import streamlit as st
from common_functions import ChainClass
from langchain_community.vectorstores import pinecone as vpc
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain.chains import RetrievalQA
import logging
from langchain.callbacks import get_openai_callback
from langchain.chains.conversation.memory import ConversationBufferMemory

    #from langchain_pinecone import PineconeVectorStore  

from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """ you assist with answering about posts that people shared on public platforms like facebook, twitter. 
in the answer, provide reference to text you understood it from. Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
MEMORY = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key='query', 
    output_key='result', 
    return_messages=True,
    max_token_limit=5000)


# Add typing for input
class Question(BaseModel):
    question: str


from pinecone import Pinecone



class RagChainClass(ChainClass):
    @retry(tries=1, delay=12)
    def get_results(self, question) -> str:
        logging.info(f"Question: {question}")
        embedding=OpenAIEmbeddings()
        query_vector = embedding.embed_query(question)

        # Query the retriever directly
        retriever = self.rag_chain.retriever
        results = retriever.get_relevant_documents(question)

        # Print details of each match
#        for match in results:
#            print(match)

        with get_openai_callback() as cb:
            embedding=OpenAIEmbeddings()
            chain_result = self.rag_chain.invoke(question, return_only_outputs=False, verbose=True)
        result = chain_result["result"]
        return(result, cb)


    @retry(tries=1, delay=12)
    def get_top_k_documents(self, question) -> str:

        logging.info(f"Question: {question}")

        # Query the retriever directly
        retriever = self.rag_chain.retriever
        results = retriever.get_relevant_documents(question)
        cb=None
        result=results
        return(result, cb)




