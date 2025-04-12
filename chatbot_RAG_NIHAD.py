
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Load environment variables
load_dotenv()

# Check for Azure OpenAI environment variables
required_env_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "COHERE_API_KEY"
]

for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} environment variable is not set")

# Define paths for document storage and vector database
DOCS_DIR = "documents"
VECTOR_DB_PATH = "vector_db"

class BankDatabaseRAGSystem:
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.chat_history = []
        self.llm = self._setup_azure_llm()
        self.embeddings = self._setup_azure_embeddings()
        
    def _setup_azure_llm(self):
        """Setup Azure OpenAI LLM"""
        return AzureChatOpenAI(
            openai_api_version="2023-05-15",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0
        )
    
    def _setup_azure_embeddings(self):
        """Setup Azure OpenAI embeddings"""
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version="2023-05-15"
        )
    
    def _generate_alternative_queries(self, query: str) -> List[str]:
        """Generate alternative phrasings of the query using LLM"""
        template = """Génère 3 façons alternatives de poser la question suivante à propos d'une base de données bancaire.
        Rends les alternatives spécifiques et concentrées sur la structure de la base de données et l'emplacement des données.
        Question originale: {question}
        
        Retourne uniquement les questions, une par ligne."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(question=query)
        
        # Split the result into individual questions
        alternative_queries = [q.strip() for q in result.split('\n') if q.strip()]
        alternative_queries.append(query)  # Include original query
        return alternative_queries
    
    def _generate_answer_explanation(self, query: str, answer: str, sources: List[Document]) -> str:
        """Generate an explanation for the answer by analyzing the sources and reasoning"""
        
        # Extract relevant content from sources
        source_texts = [doc.page_content[:300] for doc in sources[:2]]
        sources_summary = "\n".join(source_texts)
        
        template = """En tant qu'expert en base de données bancaire, explique comment tu as formulé la réponse suivante à la question posée.
        Fais référence aux données spécifiques et à la structure de la base de données bancaire.
        
        Question: {query}
        Réponse: {answer}
        
        Sources principales:
        {sources}
        
        Explique ton raisonnement en français en te concentrant sur:
        1. Comment les sources ont influencé ta réponse
        2. Les éléments clés de la structure de la base de données mentionnés
        3. Les relations entre les tables identifiées
        4. Toute implication ou considération technique importante
        
        Explication (en 3-5 phrases):"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "answer", "sources"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        explanation = chain.run(query=query, answer=answer, sources=sources_summary)
        
        return explanation.strip()
    
    def _generate_technical_commentary(self, query: str, answer: str) -> str:
        """Generate technical commentary on database structure related to the query"""
        
        template = """En tant qu'expert en bases de données bancaires, fournis un commentaire technique bref 
        sur les aspects de la structure de la base de données mentionnés dans cette question et réponse.
        
        Question: {query}
        Réponse: {answer}
        
        Fournis un commentaire technique (2-3 phrases) qui pourrait aider un développeur ou un analyste de données
        à mieux comprendre les implications techniques de cette information:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "answer"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        commentary = chain.run(query=query, answer=answer)
        
        return commentary.strip()
    
    def load_documents(self) -> List[Document]:
        """Load documents from multiple sources"""
        documents = []
        
        # Load PDF documents from directory
        if os.path.exists(os.path.join(DOCS_DIR, "pdfs")):
            pdf_loader = PyPDFDirectoryLoader(os.path.join(DOCS_DIR, "pdfs"))
            documents.extend(pdf_loader.load())
            print(f"Loaded {len(documents)} PDF documents")
        
        # Load CSV files (database schema documentation)
        if os.path.exists(os.path.join(DOCS_DIR, "csv")):
            for file in os.listdir(os.path.join(DOCS_DIR, "csv")):
                if file.endswith('.csv'):
                    csv_path = os.path.join(DOCS_DIR, "csv", file)
                    csv_loader = CSVLoader(file_path=csv_path)
                    documents.extend(csv_loader.load())
            print(f"Loaded CSV documents, total docs: {len(documents)}")
            
        # Load Excel files (might contain table structures and relationships)
        if os.path.exists(os.path.join(DOCS_DIR, "excel")):
            for file in os.listdir(os.path.join(DOCS_DIR, "excel")):
                if file.endswith(('.xlsx', '.xls')):
                    excel_path = os.path.join(DOCS_DIR, "excel", file)
                    excel_loader = UnstructuredExcelLoader(file_path=excel_path)
                    documents.extend(excel_loader.load())
            print(f"Loaded Excel documents, total docs: {len(documents)}")
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better processing"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List[Document]):
        """Create and persist vector store from documents"""
        # Check if vector store exists and load it
        if os.path.exists(VECTOR_DB_PATH):
            self.vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=self.embeddings
            )
            print(f"Loaded existing vector store with {self.vector_store._collection.count()} documents")
        else:
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=VECTOR_DB_PATH
            )
            self.vector_store.persist()
            print(f"Created new vector store with {len(documents)} documents")
    
    def setup_rag_chain(self):
        """Set up the RAG chain with reranking and custom prompts"""
        # Setup base retriever
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Setup reranker
        compressor = CohereRerank(
            model="rerank-multilingual-v2.0",  # Changed to multilingual model for French support
            api_key=os.getenv("COHERE_API_KEY"),
            top_n=4
        )
        
        # Create compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Define custom prompt template with French support
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
        
        # Create the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=compression_retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query with query reformulation and reranking
        Returns answer, sources, explanation and technical commentary"""
        if not self.chain:
            raise ValueError("RAG chain not initialized. Please run setup first.")
        
        # Generate alternative queries
        alternative_queries = self._generate_alternative_queries(query)
        
        # Process each alternative query and collect results
        all_results = []
        for alt_query in alternative_queries:
            result = self.chain({"question": alt_query, "chat_history": self.chat_history})
            all_results.append(result)
        
        # Use the result with the most relevant sources (based on reranking scores)
        best_result = max(all_results, key=lambda x: len(x["source_documents"]))
        
        # Generate explanation and commentary
        answer = best_result["answer"]
        sources = best_result["source_documents"]
        
        explanation = self._generate_answer_explanation(query, answer, sources)
        commentary = self._generate_technical_commentary(query, answer)
        
        # Update chat history with the original query and best answer
        self.chat_history.append((query, answer))
        
        return {
            "answer": answer,
            "sources": sources,
            "explanation": explanation,
            "commentary": commentary
        }

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Return the current chat history"""
        return self.chat_history

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
    
    def setup(self):
        """Complete setup process: load documents, create embeddings and chain"""
        print("Starting RAG system setup...")
        
        # Check if vector store exists
        if os.path.exists(VECTOR_DB_PATH):
            self.vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH, 
                embedding_function=self.embeddings
            )
            print(f"Loaded existing vector database from {VECTOR_DB_PATH}")
        else:
            # If not, create it
            print("Creating new vector database...")
            os.makedirs(DOCS_DIR, exist_ok=True)
            os.makedirs(os.path.join(DOCS_DIR, "pdfs"), exist_ok=True)
            os.makedirs(os.path.join(DOCS_DIR, "csv"), exist_ok=True)
            os.makedirs(os.path.join(DOCS_DIR, "excel"), exist_ok=True)
            
            documents = self.load_documents()
            if not documents:
                raise ValueError(f"No documents found in {DOCS_DIR}. Please add documents.")
                
            print(f"Processing {len(documents)} documents...")
            split_docs = self.split_documents(documents)
            print(f"Split into {len(split_docs)} chunks")
            
            self.create_vector_store(split_docs)
        
        # Setup the chain
        self.setup_rag_chain()
        print("RAG system setup complete!")

