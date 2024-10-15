#from constants import SCHEMA_IMG_PATH, LANGCHAIN_IMG_PATH
import streamlit as st
import streamlit.components.v1 as components
import json
from pyvis.network import Network
from common_functions import AddSampleQuestions
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

pinecone_api_key = st.secrets['PINECONE_API_KEY']
pc = Pinecone(api_key=pinecone_api_key)
index_name = st.secrets['PINECONE_INDEX']
index = pc.Index(index_name)
xq = embeddings.embed_query("all")
res = index.query(vector=xq, top_k=300,include_metadata=True)
entity_type_set=set()
platform_type_set=set()
post_type_set=set()
for i in (res['matches']):
    entity_type_set.add(i['metadata']['entity_type'])
    platform_type_set.add(i['metadata']['platform_type'])
    post_type_set.add(i['metadata']['post_type'])

def technical_doc_sidebar():
    with st.sidebar: 
    # Streamlit app layout
        st.title("Building Information Modeling")
        entity_type=st.sidebar.selectbox("entity_type", ['All']+list(sorted(entity_type_set)))
        platform_type=st.sidebar.selectbox("platform_type", ['All']+list(sorted(platform_type_set)))
        post_type=st.sidebar.selectbox("post_type", ['All']+list(sorted(post_type_set)))
        st.session_state["entity_type"]=entity_type
        st.session_state["platform_type"]=platform_type
        st.session_state["post_type"]=post_type
        # Example query to fetch data
        with st.sidebar:
            st.session_state["K_TOP"]= st.radio(
                "K top:",
                ("3","5","10", "20","50", "100", "200"), index=2,horizontal=True
            )
        st.session_state["MIN_COST"]= st.number_input("cost greater than: ", min_value=0, step=10000)
        # Optionally visualize graph data using third-party libraries

        sample_questions = ["summerize opinions toward arab leaders",
                            "what is the sentiment toward hizballah? quote sentences",
                            "what is the sentiment toward yihya sinwar? quote sentences "] 

        AddSampleQuestions(sample_questions)



  
