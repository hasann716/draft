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


from pinecone import Pinecone



class RagChainClass(ChainClass):
    def set_chain(self):
        self.graph_chain=None
        if "gemini" in self.model_name:
            self.rag_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key,temperature=0, verbose=True)
        else:
            self.rag_llm = ChatOpenAI(model=self.model_name, openai_api_key=self.api_key,openai_api_base=self.api_base,temperature=0)
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        index_name = 'posts-en-openai'
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)

        self.vectorstore = vpc.Pinecone(self.index,embedding=OpenAIEmbeddings(), text_key="text").\
            from_existing_index(index_name=index_name,embedding=OpenAIEmbeddings(),text_key="text")
        top_k=int(st.session_state["K_TOP"]) if "K_TOP" in st.session_state else 15
        filter={}
        if "doc_type" in st.session_state and st.session_state['doc_type'] !='All':
            filter["doc_type"]={'$eq': st.session_state["doc_type"]}
        if "entity_type" in st.session_state and st.session_state['entity_type'] !='All':
            filter["entity_type"]={'$eq': st.session_state["entity_type"]}
        if "network" in st.session_state and (st.session_state['network']) !='All':
            filter["network"]={'$eq': (st.session_state["network"])}
        self.rag_chain = RetrievalQA.from_chain_type(  
            llm=self.rag_llm,  
            chain_type="stuff",  
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k, 'filter':filter } ) ,
            memory=MEMORY
        )  

    @retry(tries=1, delay=12)
    def get_results(self, question) -> str:
        """Generate response using Pinecone Vector using vector index only

        Args:
            question (str): User query

        Returns:
            str: Formatted string answer with citations, if available.
        """
        logging.info(f"Question: {question}")
        embedding=OpenAIEmbeddings()
        query_vector = embedding.embed_query(question)

        # Query the retriever directly
        retriever = self.rag_chain.retriever
        results = retriever.get_relevant_documents(question)

        # Print details of each match
#        for match in results:
#            print(match)
        cb=None
#        with get_openai_callback() as cb:
#            embedding=OpenAIEmbeddings()
#            chain_result = self.rag_chain.invoke(question, return_only_outputs=False, verbose=True)
#            print (cb)
#        result = chain_result["result"]
        result=results
        return(result, cb)




