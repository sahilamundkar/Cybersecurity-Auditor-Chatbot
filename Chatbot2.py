import streamlit as st
import sqlite3
from datetime import datetime
import uuid

# Set page config at the very beginning
st.set_page_config(page_title="Aegis", page_icon="", layout="wide")

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
import os
from dotenv import load_dotenv
import backoff
import tiktoken

# Load environment variables
load_dotenv()

# Initialize database connection
@st.cache_resource
def init_db():
    conn = sqlite3.connect('chat_sessions.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions
                 (id TEXT PRIMARY KEY,
                  title TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  messages TEXT)''')
    
    # Check if is_active column exists, if not, add it
    c.execute("PRAGMA table_info(chat_sessions)")
    columns = [column[1] for column in c.fetchall()]
    if 'is_active' not in columns:
        c.execute("ALTER TABLE chat_sessions ADD COLUMN is_active BOOLEAN DEFAULT 1")
    
    conn.commit()
    return conn

conn = init_db()

# Function to verify API key
def verify_api_key(api_key):
    try:
        chat = ChatGroq(groq_api_key=api_key, model_name="Llama3-70b-8192")
        response = chat.invoke("Test message")
        return True
    except Exception as e:
        return False

# Initialize session state for Groq API key
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = os.environ.get('GROQ_API_KEY', '')

# Prompt for Groq API key if not set
if not st.session_state.groq_api_key:
    api_key_input = st.empty()
    new_api_key = api_key_input.text_input("Please enter your Groq API key:", type="password")
    if new_api_key:
        if verify_api_key(new_api_key):
            st.session_state.groq_api_key = new_api_key
            os.environ['GROQ_API_KEY'] = new_api_key
            api_key_input.empty()
        else:
            st.error("Invalid API key. Please check your key and try again.")
            st.stop()

# Check if API key is set before proceeding
if not st.session_state.groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

#st.image("aegis logo.JPG", use_column_width=150)

from PIL import Image
import base64

# Centered image display
st.markdown(
    """
    <style>
    .container {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

logo_path = 'aegis logo.JPG'

# Function to get the image as base64 string
def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Get the base64 string of the image
logo_base64 = get_image_as_base64(logo_path)

# Display the logo
st.markdown(
    f"""
    <div class="container">
        <img src="data:image/jpg;base64,{logo_base64}" alt="Company Logo" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize LLM with retry decorator
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def create_llm():
    return ChatGroq(groq_api_key=st.session_state.groq_api_key, model_name="Llama3-70b-8192")

llm = create_llm()

# Function to count tokens
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_prompt_with_history(history, questions_asked):
    conversation_history = "\n".join([f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in history])
    
    if questions_asked < 5:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are an AI assistant acting as an auditor to help a cybersecurity implementation engineer design the cybersecurity framework for their company using the ISO 27001 and ISO27002 standards.
            You have asked {questions_asked+1} questions so far.

            Conversation history:
            {conversation_history}

            Ask the next most appropriate question to understand the company better. Do not repeat any previous questions.
            The questions should be concise, restricted to one line and should cover only one topic.
            Format your questions as:
            Question {questions_asked+1}: [Your question here]

            Use the context and conversation history to inform your response. Provide the most accurate and relevant information possible.
            """),
            ("human", "Context: {context}\n\nUser Input: {input}")
        ])
    elif questions_asked == 5:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are an AI assistant acting as an auditor to help a cybersecurity implementation engineer design the cybersecurity framework for their company using the ISO 27001 and ISO27002 standards.
            You have asked {questions_asked+1} questions so far.

            Conversation history:
            {conversation_history}

            Based on the information provided, here are the key guidelines from ISO27001/ISO27002 for your company's cybersecurity framework:
            [Your comprehensive guidelines here, mention 10 most relevant guidelines] (While answering the guidelines, you should mention which parts/subsections/annex of which document(ISO27001 or ISO27002) you are referencing, be as descriptive as possible)
            Support your answer about each guideline by mentioning how your narrowed your search to that guideline using the information about the company (answers from the user). 

            Use the context and conversation history to inform your response. Provide the most accurate and relevant information possible.
            """),
            ("human", "Context: {context}\n\nUser Input: {input}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are an AI assistant acting as an auditor to answer the user's questions and write code if requested.

            Conversation history:
            {conversation_history}

            The user's question is the last item in the conversation.
            Please answer the user's query based on the information provided in the conversation history, the context, and your knowledge of ISO27001 and ISO27002 standards. Be specific and provide references to the relevant sections of the standards when appropriate.
            """),
            ("human", "Context: {context}\n\nUser Query: {input}")
        ])
    
    return prompt

@st.cache_resource
def load_or_create_embeddings():
    embeddings_file = "faiss_index"
    if os.path.exists(embeddings_file):
        embeddings = OllamaEmbeddings()
        vectors = FAISS.load_local(embeddings_file, embeddings, allow_dangerous_deserialization=True)
    else:
        embeddings = OllamaEmbeddings()
        loader = PyPDFDirectoryLoader("./ISO")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:20])
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local(embeddings_file)
        st.success("Created and saved new embeddings")
    return vectors

# Load embeddings at startup
vectors = load_or_create_embeddings()

# Function to save chat session
def save_chat_session(session_id, title, messages, is_active=True):
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO chat_sessions (id, title, messages, is_active) VALUES (?, ?, ?, ?)",
              (session_id, title, str(messages), is_active))
    conn.commit()

# Function to load chat session
def load_chat_session(session_id):
    c = conn.cursor()
    c.execute("SELECT title, messages FROM chat_sessions WHERE id = ?", (session_id,))
    result = c.fetchone()
    if result:
        return result[0], eval(result[1])
    return None, []

# Function to get all active chat sessions
def get_all_chat_sessions(limit=None, active_only=True):
    c = conn.cursor()
    try:
        if active_only:
            query = "SELECT id, title, created_at FROM chat_sessions WHERE is_active = ? ORDER BY created_at DESC"
            params = (1,)
        else:
            query = "SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC"
            params = ()
        
        if limit:
            query += f" LIMIT {limit}"
        
        c.execute(query, params)
    except sqlite3.OperationalError:
        # If is_active column doesn't exist, fall back to selecting all sessions
        query = "SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        c.execute(query)
    
    return c.fetchall()

# Function to delete a chat session
def delete_chat_session(session_id):
    c = conn.cursor()
    c.execute("UPDATE chat_sessions SET is_active = 0 WHERE id = ?", (session_id,))
    conn.commit()

# Function to generate a title for the chat
def generate_chat_title(messages):
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    if user_messages:
        return user_messages[0][:50] + "..."
    return "New Chat"

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "Hello! I'm Aegis, an AI Cybersecurity Auditor and an expert on ISO27001 and ISO27002 documentation. I can help you to design a cybersecurity framework for your company. Please answer the questions that follow.\n\nQuestion 1: Can you describe your company's primary business activities and industries?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 1

if "token_count" not in st.session_state:
    st.session_state.token_count = 0

if "last_reset_time" not in st.session_state:
    st.session_state.last_reset_time = time.time()

if "chat_title" not in st.session_state:
    st.session_state.chat_title = "New Chat"

def generate_response(prompt):
    conversation_history = st.session_state.messages
    prompt_template = get_prompt_with_history(conversation_history, st.session_state.questions_asked)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    try:
        response = retrieval_chain.invoke({'input': prompt, 'context': str(conversation_history)})
        reply = response['answer']
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        
        if st.session_state.questions_asked <= 5:
            st.session_state.questions_asked += 1

        with st.chat_message("assistant"):
            st.markdown(reply)
        
        st.session_state.token_count += num_tokens_from_string(reply)
        
        # Generate and update chat title after user interaction
        if len(st.session_state.messages) == 3:  # First user message
            st.session_state.chat_title = generate_chat_title(st.session_state.messages)
        
        save_chat_session(st.session_state.session_id, st.session_state.chat_title, st.session_state.messages)
    except Exception as e:
        if "rate_limit_exceeded" in str(e):
            st.warning("API rate limit reached. Please wait a moment before sending another message.")
        else:
            st.error(f"An error occurred: {str(e)}")

# Add a sidebar to display and select previous sessions
st.sidebar.title("Chat Sessions")

if st.sidebar.button("New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.questions_asked = 0
    st.session_state.chat_title = "New Chat"
    initial_message = "Hello! I'm Aegis, an AI Cybersecurity Auditor and an expert on ISO27001 and ISO27002 documentation. I can help you to design a cybersecurity framework for your company. Please answer the questions that follow.\n\nQuestion 1: Can you describe your company's primary business activities and industries?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    st.rerun()

# Display top 8 latest chat sessions
sessions = get_all_chat_sessions(limit=8)
for session in sessions:
    if st.sidebar.button(f"{session[1]}", key=f"session_{session[0]}"):
        st.session_state.session_id = session[0]
        st.session_state.chat_title, st.session_state.messages = load_chat_session(session[0])
        st.session_state.questions_asked = sum(1 for msg in st.session_state.messages if msg['role'] == 'assistant')
        st.rerun()

# Add "View All" button
if st.sidebar.button("View All Sessions"):
    st.session_state.view_all = True
    st.rerun()

# View All Sessions page
if st.session_state.get('view_all', False):
    st.title("All Chat Sessions")
    all_sessions = get_all_chat_sessions(active_only=True)
    for session in all_sessions:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"{session[1]}", key=f"all_session_{session[0]}"):
                st.session_state.session_id = session[0]
                st.session_state.chat_title, st.session_state.messages = load_chat_session(session[0])
                st.session_state.questions_asked = sum(1 for msg in st.session_state.messages if msg['role'] == 'assistant')
                st.session_state.view_all = False
                st.rerun()
        with col2:
            if st.button("Delete", key=f"delete_{session[0]}"):
                delete_chat_session(session[0])
                st.success(f"Deleted session: {session[1]}")
                st.rerun()
    if st.button("Back to Chat"):
        st.session_state.view_all = False
        st.rerun()
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Your response here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        # Check if a minute has passed since the last reset
        current_time = time.time()
        if current_time - st.session_state.last_reset_time >= 60:
            st.session_state.token_count = 0
            st.session_state.last_reset_time = current_time

        st.session_state.token_count += num_tokens_from_string(prompt)

        # Check if token count is approaching the limit (e.g., 5500 tokens to leave some buffer)
        if st.session_state.token_count > 5500:
            st.warning("Approaching token limit. The conversation history may be trimmed.")
        
        with st.spinner("Thinking..."):
            generate_response(prompt)

# Option to clear the conversation history
if st.sidebar.button("Clear Conversation History"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.questions_asked = 0
    st.session_state.chat_title = "New Chat"
    initial_message = "Hello! I'm Aegis, an AI Cybersecurity Auditor and an expert on ISO27001 and ISO27002 documentation. I can help you to design a cybersecurity framework for your company. Please answer the questions that follow.\n\nQuestion 1: Can you describe your company's primary business activities and industries?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})
    st.rerun()