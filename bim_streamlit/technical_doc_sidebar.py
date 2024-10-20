#from constants import SCHEMA_IMG_PATH, LANGCHAIN_IMG_PATH
import streamlit as st
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import json
from common_functions import AddSampleQuestions, get_meta_data_by_samples, to_epoch
from pinecone import Pinecone
import numpy as np
from streamlit_date_picker import date_range_picker, date_picker, PickerType
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
#####################################################################################
        st.markdown("### post_date range IST-2/3 ")
        default_start, default_end = datetime.now() - timedelta(days=30), datetime.now()
        refresh_value = timedelta(days=20)
        date_range_string = date_range_picker(picker_type=PickerType.date,
                                            start=default_start, end=default_end,
                                            key='date_range_picker',
                                            refresh_button={'is_show': True, 'button_name': 'Refresh Last 20 Days',
                                                            'refresh_value': refresh_value})
        if date_range_string:
            start, end = date_range_string
            st.session_state["start_datetime"]=start + "T00:00:00Z"
            st.session_state["end_datetime"]=end + "T23:59:59Z"
            st.write(f"Date Range Picker [{st.session_state["start_datetime"]}, {st.session_state["end_datetime"]}]")
            
################################################################################################        
        with st.sidebar:
            st.session_state["K_TOP"]= st.radio(
                "K top:",
                ("1", "3","5","10", "20","30", "40","50", "100", "200", "500"), index=2,horizontal=True
            )
        sample_questions = ["summerize opinions toward egypt",
                            "what is the sentiment toward hizballah? quote sentences",
                            "what is the sentiment toward hamas and their leaders? quote sentences ",
                            "what is the sentiment toward Iran?",
                            "what do people feel about lebanon government ?"] 

        AddSampleQuestions(sample_questions)



  
