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

# Add new constant for uploaded files
UPLOADS_DIR = os.path.join(os.getcwd(), "uploaded_files")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize additional session state variables for file handling
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'document_store' not in st.session_state:
    st.session_state.document_store = None

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the file path."""
    try:
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        logging.error(f"Error saving uploaded file: {str(e)}")
        raise

def process_uploaded_files(files):
    """Process multiple uploaded files and store them in ChromaDB."""
    try:
        # Initialize document store if not exists
        client = PersistentClient(path=os.path.join(UPLOADS_DIR, "document_store"))
        try:
            collection = client.get_collection(name="document_store")
            # Clear existing documents
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

        all_chunks = []
        for uploaded_file in files:
            if uploaded_file.type == "text/plain":
                # Save file and read content
                file_path = save_uploaded_file(uploaded_file)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Split content into chunks
                chunks = text_splitter.split_text(content)
                
                # Process each chunk
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
    """Retrieve relevant context from uploaded documents."""
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
            # Include source information in context
            contexts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                source = metadata.get('source', 'Unknown')
                contexts.append(f"From {source}:\n{doc}")
            return '\n\n'.join(contexts)
        return ""
    except Exception as e:
        logging.error(f"Error retrieving document context: {str(e)}")
        return ""
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

    # Initialize stores if not exists
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
                        # Combine context from knowledge base and uploaded documents
                        context = []
                        
                        if st.session_state.vectorstore is not None:
                            kb_context = get_context_from_collection(
                                st.session_state.vectorstore,
                                st.session_state.context_embeddings,
                                prompt
                            )
                            if kb_context:
                                context.append(kb_context)
                        
                        # Get context from uploaded documents
                        doc_context = get_relevant_document_context(prompt)
                        if doc_context:
                            context.append(doc_context)
                        
                        combined_context = "\n\n".join(context)

                        # Generate response
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                                  for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages, combined_context)
                        
                        # Store the interaction in chat history
                        session_id = store_chat_interaction(
                            st.session_state.chat_vectorstore,
                            st.session_state.embeddings,
                            prompt,
                            response_message,
                            session_id=st.session_state.current_session_id
                        )
                        
                        if st.session_state.current_session_id is None:
                            st.session_state.current_session_id = session_id
                        
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