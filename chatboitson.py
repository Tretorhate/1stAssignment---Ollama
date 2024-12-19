import bs4
from langchain_community.document_loaders import WebBaseLoader

bs4_strainer = bs4.SoupStrainer(class_=("content-area"))
loader = WebBaseLoader(
    web_paths=("https://pythonology.eu/using-pandas_ta-to-generate-technical-indicators-and-signals/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

local_embeddings = OllamaEmbeddings(model="all-minilm")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

question = "what are the oversold and overbought periods?"
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(question)

context = ' '.join([doc.page_content for doc in retrieved_docs])

from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2:1b")
response = llm.invoke(f"""Answer the question according to the context given very briefly:
           Question: {question}.
           Context: {context}
""")

