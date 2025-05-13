# This script implements a Streamlit-based chatbot interface (Raven) for querying laboratory regulatory information.

import os
from dotenv import load_dotenv
from pathlib import Path
import time
import sqlite3
import streamlit as st
from modules.chain import load_chain
from modules.auth import check_password

env_path = Path(__file__).parent / ".env"
success = load_dotenv(dotenv_path=env_path, override=True)
print("Looking for .env at:", env_path)

with sqlite3.connect("./sources/SQLite/textual_regulatory_data.db") as conn:
    c = conn.cursor()

# Configure Streamlit page
site_logo = 'static/logo.png'
raven_logo = 'static/raven.png'
user_logo = 'static/user.png' 
st.set_page_config(
    page_title="Raven 0.1",
    page_icon=raven_logo,
    initial_sidebar_state="expanded",
    menu_items=None
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main_app():

    # Initialize LLM chain
    chain = load_chain()

    # Initialize chat history
    if 'messages' not in st.session_state:
        # Start with first message from assistant
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Hi, I'm Raven, your laboratory regulatory assistant. Ask me about lab regulations!"},
            ]

    # Display chat messages from history on app rerun
    # Custom avatar for the assistant, default avatar for user
    for message in st.session_state.messages:
        if message["role"] == 'assistant':
            with st.chat_message(message["role"], avatar=raven_logo):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar=user_logo):
                st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input("Ask me a question!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_logo):
            st.markdown(query)

        with st.chat_message("assistant", avatar=raven_logo):
            message_placeholder = st.empty()
            # Send user's question to our chain
            result = chain({"question": query})
            response = result['answer']
            sources = [doc.metadata.get("source") for doc in result["source_documents"]]
            full_response = ""

            # Simulate streaming reponse
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Existing sidebar markdown
        st.sidebar.markdown("### Related Sources")

        # Remove duplicates by converting to a set and then back to a list
        unique_sources = list(set(sources))

        # Display markdown in the sidebar for each source
        for i, source in enumerate(unique_sources):
            # Query database for markdown content based on source
            c.execute("SELECT Content FROM regulatory_data WHERE Source = ?", (source,))
            md_content = c.fetchone()
            
            # If content exists, display it directly (no decryption)
            if md_content:
                md_content = md_content[0]
                with st.sidebar.expander(source):
                    st.markdown(md_content)
            else:
                st.sidebar.warning(f"The file for {source} could not be found.")

# Run if authenticated
if check_password():
    main_app()