#from constants import SCHEMA_IMG_PATH, LANGCHAIN_IMG_PATH
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.graphs import Neo4jGraph
import json
import networkx as nx
from pyvis.network import Network
from common_functions import AddSampleQuestions
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

pinecone_api_key = st.secrets['PINECONE_API_KEY']
pc = Pinecone(api_key=pinecone_api_key)
index_name = st.secrets['BIM_PINECONE_INDEX'] if 'BIM_PINECONE_INDEX' in st.secrets else  'posts-en-openai'
index = pc.Index(index_name)
xq = embeddings.embed_query("all")
res = index.query(vector=xq, top_k=500,include_metadata=True)

metadata_list = [item['metadata'] for item in res['matches']]

#storey_name_set=set()
#object_type_set=set()
#for i in (res['matches']):
#    storey_name_set.add(i['metadata']['storeyName'])
#    object_type_set.add(i['metadata']['objectType'])
import pandas as pd
    
meta_pddf=pd.DataFrame.from_dict(metadata_list).drop(["publish_time", "doc_text", "text"], axis=1)
numeric_max_values={}
textual_valid_values={}

keywords = ["area", "volume", "height", "depth", "length", "width", "perimeter"]
for i in (meta_pddf.columns):
    if any(keyword in i.lower() for keyword in keywords):
        numeric_max_values[i]=2*meta_pddf[i].max()
    else:
        textual_valid_values[i]=list(meta_pddf[i].unique())
def technical_objects_sidebar():
    with st.sidebar: 
    # Streamlit app layout
        st.title("BIM objects search")
        for k in textual_valid_values.keys():
            st.session_state[k]=st.sidebar.selectbox(k, ['All']+list(sorted(textual_valid_values[k])))
#        storey_name=st.sidebar.selectbox("floor name", ['All']+list(sorted(storey_name_set)))
#        object_type=st.sidebar.selectbox("object type", ['All']+list(sorted(object_type_set)))
#        age = st.slider("How old are you?", 0.0, 130.0, 0.0)
#        st.session_state["STOREY_NAME"]=storey_name
#        st.session_state["OBJECT_TYPE"]=object_type

        for k in numeric_max_values.keys():
            if (numeric_max_values[k]>0):
                st.session_state[k]=st.slider(k, 0.0, numeric_max_values[k], 0.0)



        # Example query to fetch data
        with st.sidebar:
            st.session_state["K_TOP"]= st.radio(
                "K top:",
                ("3","5","10", "20","50", "100", "200"), index=2,horizontal=True
            )
        # Optionally visualize graph data using third-party libraries

        sample_questions = ["lebanon", "עזה", "בירות" , "gaza", "west bank"]

        AddSampleQuestions(sample_questions)



  
