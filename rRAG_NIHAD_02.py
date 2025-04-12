import os
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configuration constants
DOCS_DIR = "documents"
VECTOR_DB_PATH = "vector_db"

# Global state
chat_history = []

# Environment validation
def validate_environment():
    required_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"{var} environment variable is not set")

# Azure services setup
def setup_azure_llm():
    return AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

def setup_azure_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version="2023-05-15"
    )

# Document processing functions
def load_documents() -> List[Document]:
    documents = []
    
    # PDF documents
    if os.path.exists(os.path.join(DOCS_DIR, "pdfs")):
        pdf_loader = PyPDFDirectoryLoader(os.path.join(DOCS_DIR, "pdfs"))
        documents.extend(pdf_loader.load())
    
    # CSV files
    if os.path.exists(os.path.join(DOCS_DIR, "csv")):
        for file in os.listdir(os.path.join(DOCS_DIR, "csv")):
            if file.endswith('.csv'):
                csv_path = os.path.join(DOCS_DIR, "csv", file)
                documents.extend(CSVLoader(file_path=csv_path).load())
    
    # Excel files
    if os.path.exists(os.path.join(DOCS_DIR, "excel")):
        for file in os.listdir(os.path.join(DOCS_DIR, "excel")):
            if file.endswith(('.xlsx', '.xls')):
                excel_path = os.path.join(DOCS_DIR, "excel", file)
                documents.extend(UnstructuredExcelLoader(file_path=excel_path).load())
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# Vector store management
def create_vector_store(documents: List[Document], embeddings: AzureOpenAIEmbeddings):
    if os.path.exists(VECTOR_DB_PATH):
        return Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        vector_store.persist()
        return vector_store

# RAG chain setup
def setup_rag_chain(vector_store: Chroma, llm: AzureChatOpenAI):
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    template = """Vous êtes un assistant expert en documentation de base de données bancaire.
    Votre objectif est de fournir des informations précises et spécifiques sur l'emplacement des données dans le système de base de données de la banque.
    
    Informations contextuelles issues de la documentation de la base de données ci-dessous:
    ---------------------
    {context}
    ---------------------
    
    Compte tenu des informations contextuelles et de la question, fournissez une réponse détaillée qui indique clairement:
    1. La ou les table(s) exacte(s) où se trouvent les données demandées
    2. Les noms spécifiques des colonnes
    3. Les relations avec d'autres tables
    4. Les types de données et les contraintes si pertinent
    
    Soyez précis et technique. Si les informations ne peuvent pas être trouvées dans le contexte, reconnaissez-le clairement.
    N'inventez pas d'informations sur les noms de tables ou les structures.

    Historique de la conversation: {chat_history}
    Question: {question}
    """
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=base_retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

# Query processing functions
def generate_alternative_queries(query: str, llm: AzureChatOpenAI) -> List[str]:
    template = """Génère 3 façons alternatives de poser la question suivante à propos d'une base de données bancaire.
    Rends les alternatives spécifiques et concentrées sur la structure de la base de données et l'emplacement des données.
    Question originale: {question}
    
    Retourne uniquement les questions, une par ligne."""
    
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=template, input_variables=["question"])
    )
    result = chain.run(question=query)
    return [q.strip() for q in result.split('\n') if q.strip()] + [query]

def process_query(query: str, chain: ConversationalRetrievalChain, llm: AzureChatOpenAI) -> Dict[str, Any]:
    global chat_history
    
    alternative_queries = generate_alternative_queries(query, llm)
    all_results = []
    
    for alt_query in alternative_queries:
        all_results.append(chain({"question": alt_query, "chat_history": chat_history}))
    
    best_result = max(all_results, key=lambda x: len(x["source_documents"]))
    answer = best_result["answer"]
    sources = best_result["source_documents"]
    
    # Generate explanations (implement similar functions for explanation/commentary)
    # ...
    
    chat_history.append((query, answer))
    return {
        "answer": answer,
        "sources": sources,
        # Include explanation/commentary when implemented
    }

# Main setup function
def setup_rag_system():
    validate_environment()
    
    llm = setup_azure_llm()
    embeddings = setup_azure_embeddings()
    
    if os.path.exists(VECTOR_DB_PATH):
        vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    else:
        os.makedirs(DOCS_DIR, exist_ok=True)
        documents = load_documents()
        split_docs = split_documents(documents)
        vector_store = create_vector_store(split_docs, embeddings)
    
    return {
        "llm": llm,
        "chain": setup_rag_chain(vector_store, llm),
        "vector_store": vector_store
    }
