#from constants import SCHEMA_IMG_PATH, LANGCHAIN_IMG_PATH
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
import json
from technical_doc_sidebar import technical_doc_sidebar


# Streamlit app layout
#st.title("Building Information Modeling")

# Get all secrets
models_dct = {v: k.split("_")[3] + "_API_KEY" for  k,v in st.secrets.items() if "GPT_MODEL_NAME" in k }
print (models_dct)

def ChangeButtonColour(wgt_txt, wch_hex_colour = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.color ='""" + wch_hex_colour + """'; } }</script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

def common_sidebar():
    with st.sidebar:
#        model_name=list(models_dct.keys())[0] if "GPT_MODEL_NAME" not in st.session_state else st.session_state["GPT_MODEL_NAME"] 
        page=st.sidebar.selectbox("Select a page", ["Related Posts", "RAG"])

        # Display different content based on the selected page
        model_name = st.selectbox(
            "select model- beaware: no free quote for gpt-4o!!",
            models_dct.keys(),
        )
        st.write("selected model:", model_name)
        remaining_free_queries=st.session_state["FREE_QUESTIONS_REMAINING"] if "FREE_QUESTIONS_REMAINING" in st.session_state else st.secrets["FREE_QUESTIONS_PER_SESSION"]
        st.write(f"remaining free quoata is: {remaining_free_queries} free questions")
        with st.expander(f"""Model Key- (needed after free quota is exahusted)"""):
            new_oak = st.text_input("Your API Key")
            # if "USER_OPENAI_API_KEY" not in st.session_state:
            #     st.session_state["USER_OPENAI_API_KEY"] = new_oak
            # else:
            st.session_state["USER_OPENAI_API_KEY"] = new_oak

        st.session_state["MODEL_API_KEY_TYPE"]=models_dct[model_name]
        st.session_state["GPT_MODEL_NAME"]=model_name
        if page == "RAG":
            st.session_state["USER_SELECTION"]="DOCUMENTATION"
            technical_doc_sidebar()
        elif page == "Related Posts":
            st.session_state["USER_SELECTION"]="RELATED_POSTS"
            technical_doc_sidebar()

 
