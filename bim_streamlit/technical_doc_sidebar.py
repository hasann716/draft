#from constants import SCHEMA_IMG_PATH, LANGCHAIN_IMG_PATH
import streamlit as st
import streamlit.components.v1 as components
import json
from common_functions import AddSampleQuestions, get_meta_data_by_samples
from pinecone import Pinecone
import numpy as np
random_vector = np.random.rand(1536)
pinecone_api_key = st.secrets['PINECONE_API_KEY']
pc = Pinecone(api_key=pinecone_api_key)
index_name = st.secrets['PINECONE_INDEX']
index = pc.Index(index_name)
res = index.query(vector=list(random_vector), top_k=300,include_metadata=True)
meta_dct_set, list_columns_set=get_meta_data_by_samples(res)
print(list_columns_set)
def technical_doc_sidebar():
    with st.sidebar: 
    # Streamlit app layout
        st.title("Building Information Modeling")
        for c in meta_dct_set.keys():
            st.session_state[c]=st.sidebar.selectbox(c, ['All']+list(sorted(meta_dct_set[c])))

        # Example query to fetch data
        with st.sidebar:
            st.session_state["K_TOP"]= st.radio(
                "K top:",
                ("3","5","10", "20","50", "100", "200", "500"), index=2,horizontal=True
            )
        st.session_state["MIN_COST"]= st.number_input("cost greater than: ", min_value=0, step=10000)
        # Optionally visualize graph data using third-party libraries

        sample_questions = ["summerize opinions toward arab leaders",
                            "what is the sentiment toward hizballah? quote sentences",
                            "what is the sentiment toward yihya sinwar? quote sentences "] 

        AddSampleQuestions(sample_questions)



  
