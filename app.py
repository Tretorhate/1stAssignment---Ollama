import streamlit as st
import logging
import time
import bs4
import os
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb import PersistentClient
from chromadb.errors import InvalidCollectionException
import tempfile
from langchain_core.documents import Document
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create persistent storage directories
STORAGE_DIR = os.path.join(os.getcwd(), "chat_data")
CHAT_HISTORY_DIR = os.path.join(STORAGE_DIR, "chat_history")
CONTEXT_DIR = os.path.join(STORAGE_DIR, "context")

# Create directories if they don't exist
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(CONTEXT_DIR, exist_ok=True)

# Initialize all session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_vectorstore' not in st.session_state:
    st.session_state.chat_vectorstore = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="all-minilm")
if 'context_embeddings' not in st.session_state:
    st.session_state.context_embeddings = OllamaEmbeddings(model="all-minilm")
if 'context_client' not in st.session_state:
    st.session_state.context_client = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

def migrate_chat_history(collection):
    """Migrate existing chat history to include session IDs."""
    try:
        # Get all existing documents and metadata
        results = collection.get(include=["documents", "metadatas"])
        
        if not results or not results['documents']:
            return
        
        # Retrieve document IDs separately
        document_ids = results.get('ids', [str(uuid.uuid4()) for _ in results['documents']])
        
        # Group messages created close in time (within 1 hour)
        current_time = None
        current_session = None
        updates = []
        
        for doc, metadata, doc_id in zip(results['documents'], results['metadatas'], document_ids):
            # Skip if already has session_id
            if metadata.get('session_id'):
                continue
            
            timestamp = metadata.get('timestamp')
            if not timestamp:
                timestamp = datetime.now().isoformat()
            
            try:
                msg_time = datetime.fromisoformat(timestamp)
                
                # Start new session if more than 1 hour gap or first message
                if (not current_time or 
                    not current_session or 
                    (msg_time - current_time).total_seconds() > 3600):
                    current_session = str(uuid.uuid4())
                    current_time = msg_time
                
                # Update metadata
                new_metadata = {
                    **metadata,
                    'session_id': current_session,
                    'type': 'chat_history',
                    'message_type': 'qa_pair'
                }
                
                updates.append({
                    'id': doc_id,
                    'metadata': new_metadata
                })
                
            except (ValueError, TypeError) as e:
                logging.warning(f"Error processing timestamp for document {doc_id}: {str(e)}")
                continue
        
        # Update documents in batches
        batch_size = 100
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            collection.update(
                ids=[u['id'] for u in batch],
                metadatas=[u['metadata'] for u in batch]
            )
            
        logging.info(f"Successfully migrated {len(updates)} chat history entries")
        
    except Exception as e:
        logging.error(f"Error during chat history migration: {str(e)}")

def initialize_chat_history_store():
    """Initialize ChromaDB for chat history with migration support."""
    try:
        client = PersistentClient(path=CHAT_HISTORY_DIR)
        collection = None
        
        try:
            collection = client.get_collection(name="chat_history")
            logging.info("Retrieved existing chat history collection")
            # Migrate existing data if needed
            migrate_chat_history(collection)
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
    """Store a chat interaction in ChromaDB with session tracking."""
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        chat_content = f"Q: {question}\nA: {answer}"
        embedding = embeddings.embed_query(chat_content)
        
        # Ensure timestamp is in ISO format
        current_time = datetime.now().isoformat()
        
        metadata = {
            'timestamp': current_time,
            'interaction_id': str(uuid.uuid4()),
            'session_id': session_id,
            'type': 'chat_history',
            'message_type': 'qa_pair'  # Add additional metadata for better filtering
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
    """Retrieve all unique chat sessions from ChromaDB collection."""
    try:
        if not collection:
            return []

        results = collection.get(include=["metadatas", "documents"])

        if not results or 'documents' not in results or not results['documents']:
            return []

        sessions = {}
        for doc, metadata in zip(results['documents'], results['metadatas']):
            session_id = metadata.get('session_id', str(uuid.uuid4()))  # Fallback to a new session ID
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

        # Sort sessions by last timestamp
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
    """Convert chat session messages into proper message format."""
    restored_messages = []
    for message in session_messages:
        # Split into question and answer
        parts = message.split('\nA: ')
        if len(parts) == 2:
            question = parts[0].replace('Q: ', '')
            answer = parts[1]
            restored_messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
    return restored_messages

def process_text_for_context(text):
    """Process input text into ChromaDB for context."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            add_start_index=True
        )
        
        splits = text_splitter.split_text(text)
        
        if not splits:
            raise ValueError("No text chunks were created")

        client = PersistentClient(path=CONTEXT_DIR)
        collection = None
        
        try:
            collection = client.get_collection(name="context_store")
            collection.delete(where={})
        except InvalidCollectionException:
            collection = client.create_collection(
                name="context_store",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Add documents to collection
        for i, chunk in enumerate(splits):
            embedding = st.session_state.context_embeddings.embed_query(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{'type': 'context', 'chunk_id': i}],
                ids=[f"ctx_{i}"]
            )
        
        logging.info(f"Successfully created context store with {len(splits)} chunks")
        return client, collection
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        raise

def get_context_from_collection(collection, embeddings, question, k=3):
    """Retrieve relevant context from ChromaDB collection."""
    try:
        if not collection:
            return ""
            
        query_embedding = embeddings.embed_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        if results and 'documents' in results and results['documents']:
            return ' '.join(results['documents'][0])
        return ""
    except Exception as e:
        logging.error(f"Error retrieving context: {str(e)}")
        return ""

def stream_chat(model, messages, context=None):
    """Stream chat responses with optional context enhancement."""
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        
        if context and messages:
            last_message = messages[-1]
            enhanced_content = f"""Use this context to help answer the question, but also use your general knowledge:
            Context: {context}
            
            Question: {last_message.content}"""
            messages[-1] = ChatMessage(role=last_message.role, content=enhanced_content)
        
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def clear_chat_history():
    """Clear all chat history from the database and reset session."""
    try:
        if st.session_state.chat_vectorstore:
            # Delete all chat history entries
            try:
                st.session_state.chat_vectorstore.delete(
                    where={"type": "chat_history"}
                )
            except Exception as e:
                logging.warning(f"Error with filtered delete: {str(e)}")
                # Fallback: delete all entries
                st.session_state.chat_vectorstore.delete()
            
            # Reset session state
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.success("Chat history cleared successfully!")
            logging.info("Chat history cleared")
            
            # Force refresh
            st.rerun()
    except Exception as e:
        st.error(f"Error clearing chat history: {str(e)}")
        logging.error(f"Error clearing chat history: {str(e)}")

def main():
    st.title("Enhanced Chat with LLMs")
    logging.info("App started")

    # Initialize chat history store if not exists
    if st.session_state.chat_vectorstore is None:
        try:
            client, collection = initialize_chat_history_store()
            st.session_state.chroma_client = client
            st.session_state.chat_vectorstore = collection
        except Exception as e:
            st.error(f"Failed to initialize chat history: {str(e)}")
            st.stop()

    # Sidebar configurations
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox("Choose a model", ["llama3.2"])
    
    # Add clear chat history button
    if st.sidebar.button("Clear Chat History"):
        clear_chat_history()
    
    # Context input
    st.sidebar.title("Context Management")
    context_text = st.sidebar.text_area(
        "Enter context information",
        placeholder="Paste your context text here...",
        height=300
    )
    
    if context_text and st.sidebar.button("Process Context"):
        logging.info("Processing new context text")
        with st.spinner("Processing context..."):
            try:
                client, collection = process_text_for_context(context_text)
                st.session_state.context_client = client
                st.session_state.vectorstore = collection
                st.success("Context processed successfully!")
            except Exception as e:
                st.error(f"Error processing context: {str(e)}")
                logging.error(f"Context processing error: {str(e)}")

    # Display storage location information
    st.sidebar.title("Storage Information")
    st.sidebar.info(f"""
    Data Storage Locations:
    - Chat History: {CHAT_HISTORY_DIR}
    - Context Data: {CONTEXT_DIR}
    """)

    # Display previous conversation sessions in the sidebar
# In the main() function, replace the previous conversation display section with this:

    # Display previous conversation sessions in the sidebar
    st.sidebar.title("Previous Conversations")
    if st.session_state.chat_vectorstore is not None:
        chat_sessions = get_chat_sessions(st.session_state.chat_vectorstore)
        
        # Create a container for better organization
        session_container = st.sidebar.container()
        
        # Add "New Chat" button at the top
        if session_container.button("Start New Chat", key="new_chat"):
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()
        
        # Add separator
        session_container.markdown("---")
        
        for session_id, session_data in chat_sessions:
            try:
                # Format timestamp for display
                timestamp = datetime.fromisoformat(session_data['first_timestamp'])
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
                
                # Get first message preview
                preview = ""
                if session_data['messages']:
                    first_message = session_data['messages'][0]
                    # Extract just the question part for preview
                    if "Q: " in first_message:
                        preview = first_message.split("Q: ")[1].split("\nA:")[0]
                        if len(preview) > 30:
                            preview = preview[:30] + "..."
                
                # Create a unique key using session_id
                button_key = f"session_{session_id}"
                button_label = f"{formatted_time}\n{preview}"
                
                # Create button with unique key
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
            
        # Add clear history button at the bottom
        session_container.markdown("---")
        if session_container.button("Clear All History", key="clear_history"):
            clear_chat_history()
    # Chat interface
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        # Get context from knowledge base
                        knowledge_context = ""
                        if st.session_state.vectorstore is not None:
                            knowledge_context = get_context_from_collection(
                                st.session_state.vectorstore,
                                st.session_state.context_embeddings,
                                prompt
                            )
                        
                        # Generate response
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                                  for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages, knowledge_context)
                        
                        # Store the interaction in chat history
                        session_id = store_chat_interaction(
                            st.session_state.chat_vectorstore,
                            st.session_state.embeddings,
                            prompt,
                            response_message,
                            session_id=st.session_state.current_session_id
                        )
                        
                        # Update current session ID if this is a new conversation
                        if st.session_state.current_session_id is None:
                            st.session_state.current_session_id = session_id
                        
                        # Calculate and display duration
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response generated, Duration: {duration:.2f} s")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()