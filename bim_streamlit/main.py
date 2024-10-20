import pandas as pd
from analytics import track
from free_use_manager import (
    free_questions_exhausted,
    user_supplied_openai_key_unavailable,
    decrement_free_questions,
)
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from streamlit_feedback import streamlit_feedback

import logging
#import rag_agent
from pinecone_rag_tool import PineconeRagTool
from pinecone_objects_tool import PineconeObjectsTool
import streamlit as st
from common_sidebar import common_sidebar

# Logging options
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Anonymous Session Analytics
if "SESSION_ID" not in st.session_state:
    # Track method will create and add session id to state on first run
    track("rag_demo", "appStarted", {})

# LangChain caching to reduce API calls
set_llm_cache(InMemoryCache())


image_path = "bim_streamlit/references/langchain.webp"
# Create a two-column layout
col1, col2 = st.columns([3, 1])


common_sidebar()

# Define message placeholder and emoji feedback placeholder
placeholder = st.empty()
emoji_feedback = st.empty()
user_placeholder = st.empty()
rag_tool=PineconeRagTool()
object_tool=PineconeObjectsTool()
   
# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "content": f"This is a Proof of Concept application similiarities and rag)",
        }
    ]

# Display chat messages from history on app rerun
#with placeholder.container():
#    for message in st.session_state.messages:
#        with st.chat_message(message["role"]):
#            st.markdown(message["content"], unsafe_allow_html=True)

# User input - switch between sidebar sample quick select or actual user input. Clunky but works.
#print("before free_questions_exhausted",free_questions_exhausted(), user_supplied_openai_key_unavailable() )


if "sample" in st.session_state and st.session_state["sample"] is not None:
    user_input = st.session_state["sample"]
else:
    user_input = st.chat_input(
        placeholder="Ask question ", key="user_input"
    )

if user_input:
    with user_placeholder.container():
        track("rag_demo", "question_submitted", {"question": user_input})
#        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("ai"):

            # Agent response
            with st.spinner("..."):

                message_placeholder = st.empty()
                thought_container = st.container()
                cb=None
                agent_response=None
                intermediate_steps=None
                if(st.session_state["USER_SELECTION"]=="DOCUMENTATION"):
                    agent_response, cb=rag_tool.run(tool_input=user_input)  
                    content = str(agent_response)
                    track(
                        "rag_demo", "ai_response", {"type": "rag_agent", "answer": content}
                    )
                    req_message = {"role": "user", "content": user_input}
                    resp_message = {"role": "ai", "content": content}
                    print(content)
                    with placeholder.container():
                        with st.chat_message(req_message["role"]):
                            st.markdown(req_message["content"], unsafe_allow_html=True)                        
                        with st.chat_message(resp_message["role"]):
                            st.markdown(resp_message["content"], unsafe_allow_html=True)
                elif(st.session_state["USER_SELECTION"]=="RELATED_POSTS"):
                    agent_response, cb=object_tool.run(tool_input=user_input)  
                    data = []
                    for doc in agent_response:
                        # Flatten the metadata dictionary and add it to the list
                        row = { "translation": doc.page_content  }
                        row.update(doc.metadata)
                        data.append(row)
                    # Convert the list of dictionaries to a Pandas DataFrame
                    df = pd.DataFrame(data)
                    df = df.loc[:, (df != 0).any(axis=0)]
                    df["collecting_time"]=pd.to_datetime(df['collecting_time'], unit='s')
                    df["publish_time"]=pd.to_datetime(df['publish_time'], unit='s')
                    # Display the DataFrame as a table in Streamlit
                    st.dataframe(df)
#                if isinstance(agent_response, dict) is False:
#                    agent_response = str(agent_response)
#                content = str(agent_response)
#                track(
#                    "rag_demo", "ai_response", {"type": "rag_agent", "answer": content}
#                )
#                new_message = {"role": "ai", "content": content}
#                st.session_state.messages.append(new_message)

#                st.write(f"query info: {cb}" )

#            message_placeholder.markdown(content)


    # Reinsert user chat input if sample quick select was previously used.
#    if "sample" in st.session_state and st.session_state["sample"] is not None:
#        st.session_state["sample"] = None
#        user_input = st.chat_input(
#            placeholder="Ask question on the SEC Filings", key="user_input"
#        )

#    emoji_feedback = st.empty()

# Emoji feedback
#with emoji_feedback.container():
#    feedback = streamlit_feedback(feedback_type="thumbs")
#    if feedback:
#        score = feedback["score"]
#        last_bot_message = st.session_state["messages"][-1]["content"]
#        track(
#            "rag_demo",
#            "feedback_submitted",
#            {"score": score, "bot_message": last_bot_message},
#        )
