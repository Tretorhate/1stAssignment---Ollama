import streamlit as st
import logging
import time
import bs4
import time
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
from llama_index.core import Document
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def process_text_for_context(text):
    """Process input text into a vectorstore for context."""
    try:
        # Split text into chunks using the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            add_start_index=True
        )
        
        # Split the raw text
        splits = text_splitter.split_text(text)
        
        if not splits:
            raise ValueError("No text chunks were created")

        # Create proper Langchain Document objects
        documents = [
            Document(page_content=chunk, metadata={}) 
            for chunk in splits
        ]
        
        # Update the import for OllamaEmbeddings based on deprecation warning
        from langchain_ollama import OllamaEmbeddings
        
        # Initialize embeddings and vectorstore
        embeddings = OllamaEmbeddings(model="all-minilm")
        vectorstore = Chroma.from_documents(
            documents=documents,  # Now passing proper Document objects
            embedding=embeddings,
            persist_directory=tempfile.mkdtemp()
        )
        
        logging.info(f"Successfully created vectorstore with {len(documents)} chunks")
        return vectorstore
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        raise


def get_context_from_vectorstore(vectorstore, question, k=3):
    """Retrieve relevant context from the vector store."""
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(question)
        
        # Ensure the 'text' attribute exists
        return ' '.join([getattr(doc, 'text', '') for doc in retrieved_docs if hasattr(doc, 'text')])
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
    
    # Add text input for context
    context_text = st.sidebar.text_area(
        "Enter context information",
        placeholder="Paste your context text here...",
        height=300
    )
    
    # Process context button
    if context_text and st.sidebar.button("Process Context"):
        logging.info("Processing new context text")
        with st.spinner("Processing context..."):
            try:
                st.session_state.vectorstore = process_text_for_context(context_text)
                st.success("Context processed successfully!")
            except Exception as e:
                st.error(f"Error processing context: {str(e)}")
                logging.error(f"Context processing error: {str(e)}")

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