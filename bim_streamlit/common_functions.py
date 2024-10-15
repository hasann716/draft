#from constants import SCHEMA_IMG_PATH, LANGCHAIN_IMG_PATH
import streamlit as st
import streamlit.components.v1 as components
import json
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import pinecone as vpc
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory

meta_str_filter_columns=["arena", "entity_type", "places_names", "post_type","platform_type","hashtags", "publisher_ethnicity", "publisher_gender", "entity_type", "organization_names"]
list_fields=["arena", "places_names", "hashtags", "organization_names"]
MEMORY = ConversationBufferMemory(
    memory_key="chat_history", 
    input_key='query', 
    output_key='result', 
    return_messages=True,
    max_token_limit=5000)
def ChangeButtonColour(wgt_txt, wch_hex_colour = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.color ='""" + wch_hex_colour + """'; } }</script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

def AddSampleQuestions(sample_questions):
        st.markdown("""Questions you can ask of the dataset:""", unsafe_allow_html=True)

        # To style buttons closer together
        st.markdown("""
                    <style>
                        div[data-testid="column"] {
                            width: fit-content !important;
                            flex: unset;
                        }
                        div[data-testid="column"] * {
                            width: fit-content !important;
                        }
                   </style>
                    """, unsafe_allow_html=True)
        
        for text, col in zip(sample_questions, st.columns(len(sample_questions))):
            if col.button(text, key=text):
                st.session_state["sample"] = text

class ChainClass:
    def __init__(self):
        self.api_key = st.session_state["USER_OPENAI_API_KEY"] if (("USER_OPENAI_API_KEY" in st.session_state) and (st.session_state["USER_OPENAI_API_KEY"])) else  st.secrets[st.session_state["MODEL_API_KEY_TYPE"]]
        print("api key" + self.api_key, "USER_OPENAI_API_KEY" in  st.session_state ,st.secrets[st.session_state["MODEL_API_KEY_TYPE"]])
        self.api_base=None if "GOOGLE" in st.session_state["MODEL_API_KEY_TYPE"] else st.secrets[st.session_state["MODEL_API_KEY_TYPE"].replace("KEY", "BASE")]
        self.model_name=st.session_state['GPT_MODEL_NAME']
        self.index_name=st.secrets['PINECONE_INDEX']
        self.pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        self.set_chain()

    def set_chain(self):
        self.graph_chain=None
        if "gemini" in self.model_name:
            self.rag_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key,temperature=0, verbose=True)
        else:
            self.rag_llm = ChatOpenAI(model=self.model_name, openai_api_key=self.api_key,openai_api_base=self.api_base,temperature=0)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)

        self.vectorstore = vpc.Pinecone(self.index,embedding=OpenAIEmbeddings(), text_key="text").\
            from_existing_index(index_name=self.index_name,embedding=OpenAIEmbeddings(),text_key="text")
        top_k=int(st.session_state["K_TOP"]) if "K_TOP" in st.session_state else 15
        filter={}
        for c in meta_str_filter_columns:
            if c in st.session_state and st.session_state[c] !='All':
                comparison="$eq"
                if c in list_fields:
                    comparison="$in" 
                    filter[c]={ comparison: [st.session_state[c]]}
                else:
                    filter[c]={ comparison: st.session_state[c]}

        self.rag_chain = RetrievalQA.from_chain_type(  
            llm=self.rag_llm,  
            chain_type="stuff",  
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k, 'filter':filter } ) ,
            memory=MEMORY
        )  

def get_meta_data_by_samples(res):
    meta_dct_set={}
    list_columns_set=set()
    for i in (res['matches']):
        for k in i['metadata']:
            if k in (meta_str_filter_columns):
                if k not in meta_dct_set.keys():
                    meta_dct_set[k]=set()
                if type(i['metadata'][k])!=list:
                    meta_dct_set[k].add(i['metadata'][k].lower())
                else:
                    list_columns_set.add(k)
                    for j in i['metadata'][k]:
                        meta_dct_set[k].add(j.lower())
    return(meta_dct_set, list_columns_set)