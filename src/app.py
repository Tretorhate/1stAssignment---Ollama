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

# def save_uploaded_file(uploaded_file):
#     try:
#         file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         return file_path
#     except Exception as e:
#         logging.error(f"Error saving uploaded file: {str(e)}")
#         raise


def initialize_document_store():
    try:
        client = PersistentClient(path=os.path.join(UPLOADS_DIR, "document_store"))
        
        # Try to get existing collection or create a new one
        try:
            collection = client.get_collection(name="document_store")
        except InvalidCollectionException:
            collection = client.create_collection(
                name="document_store",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Assign to session state
        st.session_state.document_store = collection
        logging.info("Document store initialized.")
    
    except Exception as e:
        logging.error(f"Error initializing document store: {str(e)}")
        st.session_state.document_store = None  # Set to None if initialization fails

# Call the function to initialize the document store at app start
if 'document_store' not in st.session_state or st.session_state.document_store is None:
    initialize_document_store()

def process_uploaded_files(files):
    """
    Process uploaded files and store them in the existing Chroma collection with embeddings.
    
    Args:
        files: List of uploaded files to process.
        
    Returns:
        int: Number of files successfully processed.
    """
    try:
        if st.session_state.document_store is None:
            logging.error("Document store is not initialized. Cannot process files.")
            return 0  # Return 0 since no files can be processed

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            add_start_index=True
        )

        processed_files = []

        for uploaded_file in files:
            if uploaded_file.type == "text/plain":
                content = uploaded_file.getvalue().decode('utf-8')
                
                chunks = text_splitter.split_text(content)
                
                # Prepare documents, embeddings, and metadata for batch addition
                documents = []
                embeddings = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(chunks):
                    embedding = st.session_state.embeddings.embed_query(chunk)
                    metadata = {
                        'source': uploaded_file.name,
                        'chunk_id': i,
                        'type': 'document',
                        'total_chunks': len(chunks)
                    }
                    documents.append(chunk)
                    embeddings.append(embedding)
                    metadatas.append(metadata)
                    ids.append(f"doc_{uploaded_file.name}_{i}")
                
                # Add all chunks to the document store
                if documents:
                    st.session_state.document_store.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                processed_files.append({
                    'name': uploaded_file.name,
                    'content_summary': content[:200] + '...' if len(content) > 200 else content
                })

        # Store information about processed files in session state for later use
        st.session_state.processed_files = processed_files
        return len(files)

    except Exception as e:
        logging.error(f"Error processing uploaded files: {str(e)}")
        raise


def get_relevant_document_context_multi_query(question, k=3):
    try:
        if st.session_state.document_store is None:
            logging.error("Document store is not initialized.")
            return "", [], []

        llm = Ollama(model="llama3.2", request_timeout=360.0)
        
        # Generate queries
        queries = generate_queries(llm, question)
        logging.info(f"Queries in retrieval function: {queries}")
        
        # Track all retrieved chunks and their ranks
        chunk_ranks = {}
        constant = 60  # RRF constant
        
        # Store all unique documents and their metadata
        doc_metadata = {}
        
        # Perform retrieval for each query
        for query_idx, query in enumerate(queries):
            query_embedding = st.session_state.embeddings.embed_query(query)
            
            results = st.session_state.document_store.query(
                query_embeddings=[query_embedding],
                n_results=k * 2,  # Retrieve more results initially for better fusion
                where={"type": "document"}
            )
            
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                
                # Calculate RRF score for each document
                for rank, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
                    # Store metadata for later use
                    doc_metadata[doc] = metadata
                    
                    if doc not in chunk_ranks:
                        chunk_ranks[doc] = 0
                    # RRF formula: 1 / (rank + constant)
                    rrf_score = 1 / (rank + constant)
                    # Weight the first query slightly higher (original query)
                    weight = 1.2 if query_idx == 0 else 1.0
                    chunk_ranks[doc] += rrf_score * weight
        
        # Sort chunks by their fused ranks
        sorted_chunks = sorted(chunk_ranks.items(), key=lambda x: x[1], reverse=True)[:k]
        
        contexts = []
        retrieved_info = []
        
        for chunk, score in sorted_chunks:
            metadata = doc_metadata[chunk]
            source = metadata.get('source', 'Unknown')
            chunk_id = metadata.get('chunk_id', 'Unknown')
            
            # Format context with source information
            context_entry = f"[Source: {source}, Chunk: {chunk_id}]\n{chunk}"
            contexts.append(context_entry)
            
            # Store detailed retrieval information
            retrieved_info.append({
                'chunk': chunk,
                'source': source,
                'chunk_id': chunk_id,
                'rrf_score': score,
                'metadata': metadata
            })

        # Join contexts with clear separators
        final_context = '\n\n---\n\n'.join(contexts)
        
        # Log retrieval statistics
        logging.info(f"Retrieved {len(sorted_chunks)} chunks from {len(queries)} queries")
        logging.info(f"Top RRF score: {sorted_chunks[0][1] if sorted_chunks else 0}")

        return final_context, queries, retrieved_info

    except Exception as e:
        logging.error(f"Error retrieving document context: {str(e)}")
        return "", [], []

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

def generate_queries(llm, user_question):
    """Generate multiple perspective queries for a given user question."""
    logging.info(f"Entering generate_queries with question: {user_question}")
    try:
        system_prompt = """You are a query generation assistant. Your task is to generate 3 alternative versions of a user's question to help with document retrieval. 
        Each version should explore different aspects or phrasings of the question.
        Return ONLY the 3 questions, one per line, with no additional text or explanation."""
        
        prompt = f"Original question: {user_question}\n\nGenerate 3 alternative versions of this question:"
        
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt)
        ]
        
        st.write("Sending to LLM:", prompt)  # Visible in the app
        response = llm.chat(messages)
        
        # Access the content directly from the response object
        response_content = response.content if hasattr(response, 'content') else str(response)
        logging.info(f"LLM Response Content: {response_content}")
        
        # Split and clean the response
        generated_queries = [q.strip() for q in response_content.strip().split('\n') if q.strip()]
        logging.info(f"Generated Queries: {generated_queries}")
        
        # Ensure we have the original question plus variants
        all_queries = [user_question]
        all_queries.extend(generated_queries)
        
        # Make sure we have at least the original question
        if len(all_queries) == 1:
            st.write("No additional queries generated, using variations of original")
            all_queries.extend([
                f"Tell me about {user_question}",
                f"What information is available about {user_question}",
                f"I want to know about {user_question}"
            ])
        
        st.write("Final queries:", all_queries)  # Visible in the app
        logging.info(f"Exiting generate_queries with queries: {all_queries}")
        return all_queries
        
    except Exception as e:
        logging.error(f"Error in generate_queries: {str(e)}")
        st.error(f"Error in generate_queries: {str(e)}")
        # Return at least the original question if there's an error
        return [user_question]

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
        # First clean and normalize the text
        text = text.replace('\r\n', '\n').strip()
        
        # Split into initial article chunks using article markers
        raw_articles = []
        current_article = []
        for line in text.split('\n'):
            if line.strip().startswith('Article'):
                if current_article:
                    raw_articles.append('\n'.join(current_article))
                current_article = [line]
            elif current_article:
                current_article.append(line)
        if current_article:
            raw_articles.append('\n'.join(current_article))
        
        client = PersistentClient(path=CONTEXT_DIR)
        
        try:
            # Get or create collection
            try:
                collection = client.get_collection(name="constitution_store")
            except InvalidCollectionException:
                collection = client.create_collection(
                    name="constitution_store",
                    metadata={"hnsw:space": "cosine"}
                )
            
            # Clear existing data
            existing_data = collection.get()
            if existing_data and 'ids' in existing_data and existing_data['ids']:
                collection.delete(ids=existing_data['ids'])
            
            # Process each article
            chunk_id = 0
            for article in raw_articles:
                if not article.strip():
                    continue
                    
                # Extract article number
                article_num = None
                if "Article" in article:
                    try:
                        article_num = article.split("Article")[1].split()[0].rstrip('.')
                        logging.info(f"Processing Article {article_num}")
                    except:
                        logging.warning("Could not extract article number")
                        continue
                
                # Create smaller chunks if article is long
                if len(article) > 500:
                    # Split while preserving sentence boundaries
                    chunk_text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=400,  # Smaller chunk size
                        chunk_overlap=50,
                        separators=["\n\n", "\n", ". ", "? ", "! "],  # Respect sentence boundaries
                        add_start_index=True
                    )
                    sub_chunks = chunk_text_splitter.split_text(article)
                else:
                    sub_chunks = [article]
                
                # Process each sub-chunk
                for i, chunk in enumerate(sub_chunks):
                    # Ensure each chunk starts with article reference if it's not there
                    if not chunk.strip().startswith('Article'):
                        chunk = f"Article {article_num} (continued):\n{chunk}"
                    
                    embedding = st.session_state.embeddings.embed_query(chunk)
                    metadata = {
                        'chunk_id': chunk_id,
                        'article_number': article_num,
                        'sub_chunk': i,
                        'total_sub_chunks': len(sub_chunks),
                        'type': 'constitution',
                        'full_article': article,  # Store complete article text
                        'is_continuation': i > 0  # Flag if this is a continuation chunk
                    }
                    
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[f"const_{article_num}_{i}"]
                    )
                    chunk_id += 1
                    
                logging.info(f"Article {article_num} split into {len(sub_chunks)} chunks")
            
            logging.info(f"Successfully processed {chunk_id} constitution chunks")
            return collection
            
        except Exception as e:
            logging.error(f"Error with collection operations: {str(e)}")
            raise
            
    except Exception as e:
        logging.error(f"Error processing constitution: {str(e)}")
        raise

def get_relevant_articles_multi_query(question, k=3):
    """Retrieve relevant constitution articles using multiple query generation."""
    try:
        if not st.session_state.constitution_store:
            logging.warning("Constitution store not initialized")
            return "", [], []  # Return empty context, queries, and retrieval info
            
        # Initialize LLM for query generation
        llm = Ollama(model="llama3.2", request_timeout=360.0)
        
        # Generate multiple queries
        queries = generate_queries(llm, question)
        
        # Track complete articles and their scores
        article_scores = {}
        retrieval_info = []
        
        for query in queries:
            query_embedding = st.session_state.embeddings.embed_query(query)
            
            results = st.session_state.constitution_store.query(
                query_embeddings=[query_embedding],
                n_results=k * 2  # Get more results initially for better coverage
            )
            
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0] if 'distances' in results else [1.0] * len(documents)
                
                for doc, metadata, score in zip(documents, metadatas, distances):
                    article_num = metadata.get('article_number')
                    if not article_num:
                        continue
                        
                    # Use the full article text if available
                    full_article = metadata.get('full_article', doc)
                    article_key = f"{article_num}:{full_article}"
                    
                    if article_key not in article_scores:
                        article_scores[article_key] = score
                        retrieval_info.append({
                            'chunk': full_article,
                            'article_number': article_num,
                            'score': score,
                            'metadata': metadata
                        })
                    else:
                        # Take the best score
                        article_scores[article_key] = min(article_scores[article_key], score)
        
        if not article_scores:
            return "No relevant articles found in the constitution. Please try rephrasing your question.", queries, []
            
        # Sort articles by score and take top k
        sorted_articles = sorted(article_scores.items(), key=lambda x: x[1])[:k]
        
        # Format context with clear article separation
        contexts = []
        for article_key, _ in sorted_articles:
            article_num, content = article_key.split(':', 1)
            contexts.append(f"Article {article_num}:\n{content.strip()}")
            
        final_context = "\n\n---\n\n".join(contexts)
        
        # Add debug information
        logging.info(f"Retrieved {len(sorted_articles)} articles")
        logging.info(f"Articles: {[key.split(':')[0] for key, _ in sorted_articles]}")
        
        return final_context, queries, retrieval_info
        
    except Exception as e:
        logging.error(f"Error retrieving articles: {str(e)}")
        return f"An error occurred while retrieving articles: {str(e)}", [], []

def stream_chat(model, messages, mode="general", context=None):
    try:
        llm = Ollama(model=model, request_timeout=360.0)
        
        if messages:
            last_message = messages[-1]
            if mode == "constitution":
                enhanced_content = f"""You are an AI assistant specialized in the Constitution of Kazakhstan. 
                Always cite specific articles when answering questions.
                
                Relevant Constitutional Articles:
                {context}
                
                Question: {last_message.content}
                
                Please provide a clear answer with specific article citations, using the information provided above."""
            else:
                enhanced_content = f"""You are a helpful AI assistant. Use this context to help answer the question, but also use your general knowledge:
                Context: {context}
                
                Question: {last_message.content}
                
                Please provide a clear and informative response, incorporating the context provided."""
            
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

def delete_document(source_name):
    """
    Delete all chunks associated with a specific document source.
    
    Args:
        source_name: Name of the document to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if st.session_state.document_store is None:
            logging.error("Document store is not initialized")
            return False

        # Get all documents with matching source
        results = st.session_state.document_store.get(
            where={"source": source_name}
        )
        
        if results and 'ids' in results and results['ids']:
            st.session_state.document_store.delete(
                ids=results['ids']
            )
            logging.info(f"Successfully deleted document: {source_name}")
            return True
        return False
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        return False

def main():
    st.title("Multi-Mode AI Assistant")
    
    # Ensure `queries` and `retrieval_info` are initialized at the very start
    queries = []  
    retrieval_info = []

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

    # Create three columns layout: sidebar (built-in), chat, and info panel
    chat_col, info_col = st.columns([2, 1])

    with st.sidebar:
        st.title("Settings")
        model = st.selectbox("Choose a model", ["llama3.2"])
        
        # File upload section
        st.title("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload text documents",
            type=['txt'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        num_files = process_uploaded_files(uploaded_files)
                        st.success(f"Successfully processed {num_files} documents!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")

        # Display uploaded documents
        if st.session_state.document_store:
            st.title("Uploaded Documents")
            try:
                results = st.session_state.document_store.get(
                    where={"type": "document"},
                    include=["metadatas"]
                )
                if results and results['metadatas']:
                    unique_sources = set(meta['source'] for meta in results['metadatas'])
                    st.write("Available documents:")
                    for source in unique_sources:
                        col1, col2 = st.columns([3, 1])
                        col1.text(f"📄 {source}")
                        if col2.button("🗑️", key=f"delete_{source}", help=f"Delete {source}"):
                            if delete_document(source):
                                st.success(f"Deleted {source}")
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {source}")
            except Exception as e:
                st.error(f"Error displaying documents: {str(e)}")

        # Previous conversations section
        st.title("Previous Conversations")
        if st.session_state.chat_vectorstore is not None:
            chat_sessions = get_chat_sessions(st.session_state.chat_vectorstore)
            
            session_container = st.container()
            
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

    # Main chat interface in left column
    with chat_col:
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input and response
        chat_placeholder = "Ask anything" if st.session_state.assistant_mode == "general" else "Ask about the Constitution of Kazakhstan"
        
        if prompt := st.chat_input(chat_placeholder):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..." if st.session_state.assistant_mode == "general" else "Researching Constitution..."):
                    try:
                        context = ""
                        
                        if st.session_state.assistant_mode == "constitution":
                            context, queries, retrieval_info = get_relevant_articles_multi_query(prompt)
                        else:
                            context, queries, retrieval_info = get_relevant_document_context_multi_query(prompt)
                            logging.info(f"Retrieved all: {context, queries, retrieval_info}")
                        
                        # Add debug output
                        st.write("Debug - Retrieval Info:", retrieval_info)
                        
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                                  for msg in st.session_state.messages]
                        response = stream_chat(model, messages, st.session_state.assistant_mode, context)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Store the interaction
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

    # Info panel in right column
    with info_col:
        st.write("Debug - Current Mode:", st.session_state.assistant_mode)
        
        # Display the queries generated
        if queries:
            with st.expander("Generated Queries", expanded=True):
                st.markdown("### Generated Queries")
                for i, query in enumerate(queries, 1):
                    st.markdown(f"**Query {i}:** {query}")
        else:
            st.warning("No queries generated.")


        # Display retrieved chunks if available
        # if retrieval_info and 'chunks' in retrieval_info:
        #     with st.expander("Retrieved Chunks", expanded=True):
        #         st.markdown("### Retrieved Content")
        #         for i, info in enumerate(retrieval_info['chunks'], 1):
        #             st.markdown(f"**Chunk {i}** (Source: {info['source']})")
        #             st.markdown(f"Relevance: {1 - info['score']:.3f}")
        #             st.text(info['chunk'])
        #             st.markdown("---")

if __name__ == "__main__":
    main()