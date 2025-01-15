import streamlit as st
import logging
import time
import os
import uuid
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import shutil
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb import PersistentClient
from chromadb.errors import InvalidCollectionException

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create storage directories
STORAGE_DIR = os.path.join(os.getcwd(), "ai_assistant_data")
CHAT_HISTORY_DIR = os.path.join(STORAGE_DIR, "chat_history")
CONTEXT_DIR = os.path.join(STORAGE_DIR, "context")
UPLOADS_DIR = os.path.join(os.getcwd(), "uploaded_files")

# Create directories
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(CONTEXT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_vectorstore' not in st.session_state:
    st.session_state.chat_vectorstore = None
if 'document_store' not in st.session_state:
    st.session_state.document_store = None
if 'constitution_store' not in st.session_state:
    st.session_state.constitution_store = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="all-minilm")
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'assistant_mode' not in st.session_state:
    st.session_state.assistant_mode = "general"
if 'constitution_loaded' not in st.session_state:
    st.session_state.constitution_loaded = False

# Helper functions

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        logging.error(f"Error saving uploaded file: {str(e)}")
        raise

def process_uploaded_files(files):
    try:
        client = PersistentClient(path=os.path.join(UPLOADS_DIR, "document_store"))
        try:
            collection = client.get_collection(name="document_store")
            collection.delete(where={})
        except InvalidCollectionException:
            collection = client.create_collection(
                name="document_store",
                metadata={"hnsw:space": "cosine"}
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            add_start_index=True
        )

        for uploaded_file in files:
            if uploaded_file.type == "text/plain":
                file_path = save_uploaded_file(uploaded_file)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    embedding = st.session_state.embeddings.embed_query(chunk)
                    metadata = {
                        'source': uploaded_file.name,
                        'chunk_id': i,
                        'type': 'document'
                    }
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[f"doc_{uploaded_file.name}_{i}"]
                    )

        st.session_state.document_store = collection
        return len(files)
    except Exception as e:
        logging.error(f"Error processing uploaded files: {str(e)}")
        raise

def get_relevant_document_context(question, k=3):
    try:
        if not st.session_state.document_store:
            return ""

        query_embedding = st.session_state.embeddings.embed_query(question)
        results = st.session_state.document_store.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"type": "document"}
        )

        if results and 'documents' in results and results['documents']:
            contexts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                source = metadata.get('source', 'Unknown')
                contexts.append(f"From {source}:\n{doc}")
            return '\n\n'.join(contexts)
        return ""
    except Exception as e:
        logging.error(f"Error retrieving document context: {str(e)}")
        return ""

def initialize_chat_history_store():
    try:
        client = PersistentClient(path=CHAT_HISTORY_DIR)
        try:
            collection = client.get_collection(name="chat_history")
            logging.info("Retrieved existing chat history collection")
        except InvalidCollectionException:
            collection = client.create_collection(
                name="chat_history",
                metadata={"hnsw:space": "cosine"}
            )
            logging.info("Created new chat history collection")
        
        return client, collection
    except Exception as e:
        logging.error(f"Error initializing chat history store: {str(e)}")
        raise

def store_chat_interaction(collection, embeddings, question, answer, session_id=None):
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        chat_content = f"Q: {question}\nA: {answer}"
        embedding = embeddings.embed_query(chat_content)
        
        current_time = datetime.now().isoformat()
        
        metadata = {
            'timestamp': current_time,
            'interaction_id': str(uuid.uuid4()),
            'session_id': session_id,
            'type': 'chat_history',
            'message_type': 'qa_pair'
        }
        
        collection.add(
            documents=[chat_content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )
        return session_id
    except Exception as e:
        logging.error(f"Error storing chat interaction: {str(e)}")
        raise

def get_chat_sessions(collection):
    try:
        if not collection:
            return []

        results = collection.get(include=["metadatas", "documents"])

        if not results or 'documents' not in results or not results['documents']:
            return []

        sessions = {}
        for doc, metadata in zip(results['documents'], results['metadatas']):
            session_id = metadata.get('session_id', str(uuid.uuid4()))
            timestamp = metadata.get('timestamp', datetime.now().isoformat())

            if session_id not in sessions:
                sessions[session_id] = {
                    'messages': [],
                    'first_timestamp': timestamp,
                    'last_timestamp': timestamp
                }

            sessions[session_id]['messages'].append(doc)

            try:
                current_last = datetime.fromisoformat(sessions[session_id]['last_timestamp'])
                new_timestamp = datetime.fromisoformat(timestamp)
                if new_timestamp > current_last:
                    sessions[session_id]['last_timestamp'] = timestamp
            except ValueError:
                continue

        try:
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: datetime.fromisoformat(x[1]['last_timestamp']),
                reverse=True
            )
        except ValueError:
            sorted_sessions = list(sessions.items())

        return sorted_sessions
    except Exception as e:
        logging.error(f"Error retrieving chat sessions: {str(e)}")
        return []

def restore_chat_session(session_messages):
    restored_messages = []
    for message in session_messages:
        parts = message.split('\nA: ')
        if len(parts) == 2:
            question = parts[0].replace('Q: ', '')
            answer = parts[1]
            restored_messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
    return restored_messages

def clear_chat_history():
    try:
        if st.session_state.chat_vectorstore:
            try:
                st.session_state.chat_vectorstore.delete(
                    where={"type": "chat_history"}
                )
            except Exception as e:
                logging.warning(f"Error with filtered delete: {str(e)}")
                st.session_state.chat_vectorstore.delete()
            
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.success("Chat history cleared successfully!")
            logging.info("Chat history cleared")
            
            st.rerun()
    except Exception as e:
        st.error(f"Error clearing chat history: {str(e)}")
        logging.error(f"Error clearing chat history: {str(e)}")

def fetch_constitution():
    url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('main') or soup.find('article')
        if not content:
            logging.warning("Could not find main content section")
            return ""
        return content.get_text()
    except Exception as e:
        logging.error(f"Error fetching constitution: {str(e)}")
        return ""

def process_constitution_text(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\nArticle", "\n\nSection", "\n\n"],
            add_start_index=True
        )
        
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise ValueError("No text chunks were created")
        
        client = PersistentClient(path=CONTEXT_DIR)
        
        try:
            try:
                collection = client.get_collection(name="constitution_store")
                collection.delete(where={})
                logging.info("Cleared existing constitution collection")
            except Exception:
                collection = client.create_collection(
                    name="constitution_store",
                    metadata={"hnsw:space": "cosine"}
                )
                logging.info("Created new constitution collection")
            
            for i, chunk in enumerate(chunks):
                article_num = None
                if "Article" in chunk:
                    try:
                        article_num = chunk.split("Article")[1].split()[0].rstrip('.')
                    except:
                        pass
                    
                embedding = st.session_state.embeddings.embed_query(chunk)
                metadata = {
                    'chunk_id': i,
                    'article_number': article_num,
                    'type': 'constitution'
                }
                
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[f"const_{i}"]
                )
            
            logging.info(f"Successfully processed {len(chunks)} constitution chunks")
            return collection
            
        except Exception as e:
            logging.error(f"Error with collection operations: {str(e)}")
            raise
            
    except Exception as e:
        logging.error(f"Error processing constitution: {str(e)}")
        raise

def get_relevant_articles(question, k=3):
    try:
        if not st.session_state.constitution_store:
            logging.warning("Constitution store not initialized")
            return "Constitution data is not available. Please try reloading the Constitution mode."
            
        query_embedding = st.session_state.embeddings.embed_query(question)
        results = st.session_state.constitution_store.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        if results and 'documents' in results and results['documents']:
            contexts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                article_num = metadata.get('article_number', 'Section')
                contexts.append(f"Article {article_num}:\n{doc}")
            return '\n\n'.join(contexts)
        return "No relevant articles found."
    except Exception as e:
        logging.error(f"Error retrieving articles: {str(e)}")
        return f"An error occurred while retrieving articles: {str(e)}"

def stream_chat(model, messages, mode="general", context=None):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        
        if messages:
            last_message = messages[-1]
            if mode == "constitution":
                enhanced_content = f"""You are an AI assistant specialized in the Constitution of Kazakhstan. 
                Always cite specific articles when answering questions.
                
                Relevant Constitutional Articles:
                {context}
                
                Question: {last_message.content}
                
                Please provide a clear answer with specific article citations."""
            else:
                enhanced_content = f"""You are a helpful AI assistant. Use this context to help answer the question, but also use your general knowledge:
                Context: {context}
                
                Question: {last_message.content}
                
                Please provide a clear and informative response."""
            
            messages[-1] = ChatMessage(role=last_message.role, content=enhanced_content)
        
        response = ""
        response_placeholder = st.empty()
        
        for chunk in llm.stream_chat(messages):
            response += chunk.delta
            response_placeholder.write(response)
            
        return response
    except Exception as e:
        logging.error(f"Error during chat: {str(e)}")
        raise

def initialize_constitution_mode():
    try:
        if not st.session_state.constitution_loaded:
            with st.spinner("Loading Constitution..."):
                constitution_text = fetch_constitution()
                if not constitution_text:
                    st.error("Failed to fetch Constitution text. Please check your internet connection.")
                    return False
                
                try:
                    st.session_state.constitution_store = process_constitution_text(constitution_text)
                    if st.session_state.constitution_store is None:
                        raise ValueError("Failed to process Constitution text")
                    st.session_state.constitution_loaded = True
                    st.success("Constitution loaded successfully!")
                    return True
                except Exception as e:
                    st.error(f"Error processing Constitution: {str(e)}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Error in constitution mode initialization: {str(e)}")
        st.error("Failed to initialize Constitution mode. Please try again.")
        return False

def main():
    st.title("Multi-Mode AI Assistant")
    
    # Initialize stores if not exists
    if st.session_state.chat_vectorstore is None:
        try:
            client, collection = initialize_chat_history_store()
            st.session_state.chat_vectorstore = collection
        except Exception as e:
            st.error(f"Failed to initialize chat history: {str(e)}")
            st.stop()
    
    # Assistant mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("General Assistant", 
                    type="primary" if st.session_state.assistant_mode == "general" else "secondary",
                    key="general_button"):
            st.session_state.assistant_mode = "general"
            st.session_state.messages = []
            st.rerun()
            
    with col2:
        if st.button("Constitution Assistant", 
                    type="primary" if st.session_state.assistant_mode == "constitution" else "secondary",
                    key="constitution_button"):
            st.session_state.assistant_mode = "constitution"
            st.session_state.messages = []
            if initialize_constitution_mode():
                st.rerun()

    # Display current mode
    if st.session_state.assistant_mode:
        st.markdown(f"**Current Mode:** {st.session_state.assistant_mode.title()} Assistant")
        st.markdown("---")
    else:
        st.info("Please select an assistant mode above to begin.")
        st.stop()

    # Sidebar settings
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox("Choose a model", ["llama3.2"])
    
    # File upload section in sidebar
    st.sidebar.title("Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload text documents",
        type=['txt'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.sidebar.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    num_files = process_uploaded_files(uploaded_files)
                    st.success(f"Successfully processed {num_files} documents!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

    # Display uploaded files
    if st.session_state.document_store:
        st.sidebar.title("Uploaded Documents")
        try:
            results = st.session_state.document_store.get(
                where={"type": "document"},
                include=["metadatas"]
            )
            if results and results['metadatas']:
                unique_sources = set(meta['source'] for meta in results['metadatas'])
                st.sidebar.write("Available documents:")
                for source in unique_sources:
                    st.sidebar.text(f"📄 {source}")
        except Exception as e:
            st.sidebar.error(f"Error displaying documents: {str(e)}")

    # Display previous conversation sessions
    st.sidebar.title("Previous Conversations")
    if st.session_state.chat_vectorstore is not None:
        chat_sessions = get_chat_sessions(st.session_state.chat_vectorstore)
        
        session_container = st.sidebar.container()
        
        if session_container.button("Start New Chat", key="new_chat"):
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()
        
        session_container.markdown("---")
        
        for session_id, session_data in chat_sessions:
            try:
                timestamp = datetime.fromisoformat(session_data['first_timestamp'])
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
                
                preview = ""
                if session_data['messages']:
                    first_message = session_data['messages'][0]
                    if "Q: " in first_message:
                        preview = first_message.split("Q: ")[1].split("\nA:")[0]
                        if len(preview) > 30:
                            preview = preview[:30] + "..."
                
                button_key = f"session_{session_id}"
                button_label = f"{formatted_time}\n{preview}"
                
                if session_container.button(
                    button_label,
                    key=button_key,
                    help=f"Click to restore this conversation"
                ):
                    restored_messages = restore_chat_session(session_data['messages'])
                    st.session_state.messages = restored_messages
                    st.session_state.current_session_id = session_id
                    st.rerun()
                
            except Exception as e:
                logging.error(f"Error displaying session {session_id}: {str(e)}")
                continue
        
        session_container.markdown("---")
        if session_container.button("Clear All History", key="clear_history"):
            clear_chat_history()

    #Added code starts here
    if st.session_state.assistant_mode == "constitution":
        if st.sidebar.button("Reload Constitution Data"):
            st.session_state.constitution_loaded = False
            if initialize_constitution_mode():
                st.success("Constitution data reloaded successfully!")
            else:
                st.error("Failed to reload Constitution data. Please try again.")
    #Added code ends here

    # Chat interface
    chat_placeholder = "Ask anything" if st.session_state.assistant_mode == "general" else "Ask about the Constitution of Kazakhstan"
    
    if prompt := st.chat_input(chat_placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..." if st.session_state.assistant_mode == "general" else "Researching Constitution..."):
                    try:
                        context = ""
                        if st.session_state.assistant_mode == "constitution":
                            context = get_relevant_articles(prompt)
                        else:
                            context = get_relevant_document_context(prompt)
                        
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                                  for msg in st.session_state.messages]
                        response = stream_chat(model, messages, st.session_state.assistant_mode, context)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Store the interaction in chat history
                        session_id = store_chat_interaction(
                            st.session_state.chat_vectorstore,
                            st.session_state.embeddings,
                            prompt,
                            response,
                            session_id=st.session_state.current_session_id
                        )
                        
                        if st.session_state.current_session_id is None:
                            st.session_state.current_session_id = session_id

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

