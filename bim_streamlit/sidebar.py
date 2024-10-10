#from constants import SCHEMA_IMG_PATH, LANGCHAIN_IMG_PATH
import streamlit as st
import streamlit.components.v1 as components
import json
from pyvis.network import Network
from common_functions import AddSampleQuestions



# Example query to fetch data

# Optionally visualize graph data using third-party libraries


def sidebar():
    with st.sidebar: 
#        st.title("Neo4j Graph Visualization with Relationships")
        # Streamlit app
        sample_questions = "How many storeys in the building?", "what materials are used?", "get distinct object names that have volume",\
            "what is the total slab volume by floor", "כמה קירות יש בבניין?","כמה קורות יש בקומה 15",\
            "מהן שתי הקומות הראשונות?","מה נפח הקורות הכולל בבניין?", "מה נפח הרצפה הכולל?","What is the total length of the column in the building?", \
            "מהי הקומה האמצעית בבניין?", "מה נפח הקורות בקומה הגבוהה ביותר? מה שמה ? ומה גובהה?", "מהו סכום שטח הרצפה הכולל של רצפות עבות מ25 סנטטמטר בקומה 5"

        AddSampleQuestions(sample_questions)


  
