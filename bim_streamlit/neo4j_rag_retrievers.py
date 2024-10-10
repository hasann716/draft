from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Typical RAG retriever

typical_rag = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(), index_name="typical_rag", url=st.secrets["DOC_NEO4J_URI"],
    username=st.secrets["DOC_NEO4J_USERNAME"],
    password=st.secrets["DOC_NEO4J_PASSWORD"]
)

# Parent retriever

parent_query = """
MATCH (node)<-[:HAS_CHILD]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata LIMIT 1
"""

parent_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    index_name="parent_document",
    retrieval_query=parent_query,url=st.secrets["DOC_NEO4J_URI"],
    username=st.secrets["DOC_NEO4J_USERNAME"],
    password=st.secrets["DOC_NEO4J_PASSWORD"]
)

# Hypothetic questions retriever

hypothetic_question_query = """
MATCH (node)<-[:HAS_QUESTION]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    index_name="hypothetical_questions",
    retrieval_query=hypothetic_question_query,url=st.secrets["DOC_NEO4J_URI"],
    username=st.secrets["DOC_NEO4J_USERNAME"],
    password=st.secrets["DOC_NEO4J_PASSWORD"]
)
# Summary retriever

summary_query = """
MATCH (node)<-[:HAS_SUMMARY]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

summary_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(),
    index_name="summary",
    retrieval_query=summary_query,url=st.secrets["DOC_NEO4J_URI"],
    username=st.secrets["DOC_NEO4J_USERNAME"],
    password=st.secrets["DOC_NEO4J_PASSWORD"]
)
