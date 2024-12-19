import streamlit as st
import logging
import time
import bs4
import time
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import OllamaLLM

logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def load_and_process_url(url, class_name=None):
    max_retries = 3
    retry_delay = 1  # in seconds

    for attempt in range(max_retries):
        try:
            # Configure BeautifulSoup strainer if class name is provided
            bs4_strainer = bs4.SoupStrainer(class_=(class_name,)) if class_name else None
            loader_kwargs = {"bs_kwargs": {"parse_only": bs4_strainer}} if bs4_strainer else {}
            
            # Load the webpage
            loader = WebBaseLoader(web_paths=(url,), **loader_kwargs)
            docs = loader.load()
            
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100,
                add_start_index=True
            )
            splits = text_splitter.split_documents(docs)
            
            # Create embeddings and vectorstore
            embeddings = OllamaEmbeddings(model="all-minilm") # llama3.2 all-minilm
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            return vectorstore
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt+1} failed with error: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"All retries failed with error: {str(e)}")
                raise e

def get_context_from_vectorstore(vectorstore, question, k=3):
    """Retrieve relevant context from the vector store."""
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(question)
        return ' '.join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        logging.error(f"Error retrieving context: {str(e)}")
        return ""

def stream_chat(model, messages, context=None):
    """Stream chat responses with optional context enhancement."""
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        
        # If context is provided, enhance the last message with it
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

def main():
    st.title("Enhanced Chat with LLMs")
    logging.info("App started")

    # Sidebar configurations
    model = st.sidebar.selectbox("Choose a model", ["llama3.2"])
    url = st.sidebar.text_input("Enter URL for context (optional)")
    class_name = st.sidebar.text_input("Enter class name for web scraping (optional)")
    
    # Load URL data if provided
    if url and st.sidebar.button("Load URL"):
        with st.spinner("Processing URL..."):
            try:
                st.session_state.vectorstore = load_and_process_url(url, class_name)
                st.success("URL processed successfully!")
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")

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
                        # Get context if vectorstore exists
                        context = None
                        if st.session_state.vectorstore:
                            context = get_context_from_vectorstore(
                                st.session_state.vectorstore,
                                prompt
                            )

                        # Generate response
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) 
                                  for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages, context)
                        
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