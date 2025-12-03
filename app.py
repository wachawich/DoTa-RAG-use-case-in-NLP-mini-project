import streamlit as st
import os
from dotenv import load_dotenv

from cohere import Client as CohereClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

from search import rag_search
from QA import ask

# --------------------------
# INITIAL LOAD
# --------------------------
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))
index = pc.Index("wiki-embeddings")

co = CohereClient(os.getenv("COHERE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

sentence_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")


# --------------------------
# STREAMLIT PAGE CONFIG
# --------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Wikipedia QA")


# --------------------------
# CHAT HISTORY STATE
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": user/assistant, "content": text}


# --------------------------
# CHAT MESSAGE DISPLAY
# --------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# --------------------------
# USER INPUT (ChatGPT style)
# --------------------------
user_input = st.chat_input("Type your question...")

if user_input:
    # add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # display user message
    with st.chat_message("user"):
        st.write(user_input)

    # BOT RESPONSE
    with st.chat_message("assistant"):
        # with st.spinner("Processing RAG..."):
        with st.status("Processing your question...", expanded=True) as status:

            # 1) RAG search
            result = rag_search(
                groq_client,
                sentence_model,
                index,
                co,
                question=user_input,
                ns_top_k=4,
                per_namespace=25,
                final_k=10
            )

            print(result)
            # 2) LLM answer
            if len(result) == 0:
                st.write("Step 3: Ask with AI.")
            else :
                st.write("Step 5: Ask with AI.")
            response = ask(groq_client, result, user_input)
            
            status.update(label="✨ Completed!", state="complete")

        st.write(response)

    # save bot reply to history
    st.session_state.messages.append({"role": "assistant", "content": response})
