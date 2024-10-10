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
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain.chains import RetrievalQA
import logging
from langchain.callbacks import get_openai_callback
from langchain.chains.conversation.memory import ConversationBufferMemory

from neo4j_rag_retrievers import (
    hypothetic_question_vectorstore,
    parent_vectorstore,
    summary_vectorstore,
    typical_rag,
)
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


template = """Answer the question based only on the following context:
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
retrieval_query_dict={
    "typical_rag": None,
    "parent_document": """
                    MATCH (node)<-[:HAS_CHILD]-(parent)
                    WITH parent, max(score) AS score // deduplicate parents
                    RETURN parent.text AS text, score, {} AS metadata LIMIT 1
                    """,
    "hypothetical_questions": """
                    MATCH (node)<-[:HAS_QUESTION]-(parent)
                    WITH parent, max(score) AS score // deduplicate parents
                    RETURN parent.text AS text, score, {} AS metadata
                    """,
    "summary" : """
        MATCH (node)<-[:HAS_SUMMARY]-(parent)
        WITH parent, max(score) AS score // deduplicate parents
        RETURN parent.text AS text, score, {} AS metadata
        """

}
class RagChainClass(ChainClass):
    def set_chain(self):
        print("setting new graphchain")
        print(self.model_name, self.api_base, self.api_key)
        self.graph_chain=None
        if "gemini" in self.model_name:
            self.rag_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key,temperature=0, verbose=True,top_k=200)
        else:
            self.rag_llm = ChatOpenAI(model=self.model_name, openai_api_key=self.api_key,openai_api_base=self.api_base,temperature=0)
        index_name=st.session_state["RAG_STRATEGY"] if "RAG_STRATEGY" in st.session_state else "hypothetical_questions"
        self.vectorstore=Neo4jVector.from_existing_index(
        OpenAIEmbeddings(), index_name=index_name, url=st.secrets["DOC_NEO4J_URI"],
        username=st.secrets["DOC_NEO4J_USERNAME"],
        password=st.secrets["DOC_NEO4J_PASSWORD"], retrieval_query=None)
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.rag_llm, chain_type="stuff"
            , retriever=self.vectorstore.as_retriever(),
            memory=MEMORY,
        )      
        
    @retry(tries=1, delay=12)
    def get_results(self, question) -> str:
        """Generate response using Neo4jVector using vector index only

        Args:
            question (str): User query

        Returns:
            str: Formatted string answer with citations, if available.
        """
        logging.info(f"Question: {question}")
        with get_openai_callback() as cb:
            chain_result = self.rag_chain.invoke(question, return_only_outputs=True)
            print (cb)
        logging.debug(f"chain_result: {chain_result}")
        result = chain_result["result"]
        return(result, cb)




