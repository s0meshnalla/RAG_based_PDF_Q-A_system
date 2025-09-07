from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import sys
import os

load_dotenv()

# Step 1: Loading PDFs from a directory
try:
    loader = DirectoryLoader(
        path="data",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    data = loader.load()
except FileNotFoundError:
    print("Error: The given directory does not exist. Please create it and add PDF files.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading PDFs: {str(e)}")
    sys.exit(1)

# Step 2: Check if documents were loaded and spliting it into chunks
if not data:
    print("Error: No documents were loaded from the given directory. Please check the directory and ensure it contains valid PDF files.")
    sys.exit(1)

try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    print(f"Total number of documents: {len(docs)}")
except Exception as e:
    print(f"Error splitting documents: {str(e)}")
    sys.exit(1)

# Step 3: Embeddings + Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# Step 4: Retrieving chunks
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Step 5: defining Ollama LLM
chat_llm = ChatOllama(
    model="mistral",
    temperature=0.5,
    model_kwargs={"num_predict": 512}
)

# Step 6: RAG Chain implementation
system_prompt = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say you don't know.
If the user greets you, respond with a greeting.

{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chat_llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Step 7: Query loop
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    response = rag_chain.invoke({"input": query})
    print("\nAnswer:", response["answer"])

    print("\nRetrieved Chunks Used for Answering:\n")
    for i, doc in enumerate(response.get("context", []), 1):
        doc_name = doc.metadata.get("source", "Unknown document")
        doc_name = os.path.basename(doc_name)
        print(f"[{i}] Document: {doc_name}")
        print(f"Content: {doc.page_content.strip()[:500]}...\n")