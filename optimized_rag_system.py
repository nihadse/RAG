import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from datetime import datetime
import json
import os
from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib

# Azure OpenAI imports
from openai import AzureOpenAI
import httpx

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Centralized configuration management"""
    
    # Azure OpenAI Configuration
    OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID", "your_client_id")
    OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET", "your_client_secret")
    OIDC_ENDPOINT = os.getenv("OIDC_ENDPOINT", "https://alfactory.api.staging.schonet/auth/oauth2/v2/token")
    OIDC_SCOPE = os.getenv("OIDC_SCOPE", "genai-model")
    
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://alfactory.api.staging.schonet/genai-model/v1")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "FAKE_KEY")
    AZURE_MODEL_DEPLOYMENT = os.getenv("AZURE_MODEL_DEPLOYMENT", "gpt-4o")
    
    # Embedding Configuration
    EMBEDDING_MODEL_PATH = "/domino/datasets/local/test-prd-base"
    EMBEDDING_DIMENSION = 1024
    
    # ChromaDB Configuration
    CHROMA_PERSIST_PATH = "/domino/datasets/local/chroma_persistent_db"
    COLLECTION_NAME = "banking_documents"
    
    # Chunking Configuration for Banking Documents
    CHUNK_SIZES = {
        'header': 300,      # For sections with headers
        'table': 500,       # For tables and structured data
        'paragraph': 400,   # For regular paragraphs
        'list': 350,        # For lists and enumerations
    }
    CHUNK_OVERLAP = 100
    
    # BM25 Configuration
    BM25_INDEX_PATH = "/domino/datasets/local/bm25_index.pkl"
    BM25_K1 = 1.5
    BM25_B = 0.75
    
    # Retrieval Configuration
    INITIAL_RETRIEVAL_K = 20  # Retrieve more for reranking
    FINAL_RESULTS_K = 5       # Final results after reranking
    HYBRID_ALPHA = 0.5        # Balance between semantic (0) and keyword (1)
    
    # Batch Processing Configuration
    BATCH_SIZE = 100
    MAX_WORKERS = 4

# ==============================================================================
# DOCUMENT PROCESSING & CHUNKING
# ==============================================================================

class BankingDocumentProcessor:
    """Advanced document processor optimized for banking/technical documents"""
    
    def __init__(self):
        self.chunk_patterns = {
            'section_header': re.compile(r'^(Sect\.|Section|Chapitre|Article|§)\s*[\dIVX]+[\.\s]', re.IGNORECASE),
            'subsection': re.compile(r'^[A-Z]{1,3}\.\d+(\.\d+)*\s+', re.MULTILINE),
            'table_marker': re.compile(r'(Description Rubrique|Num\s+donn\.|Val\s+init)', re.IGNORECASE),
            'code_block': re.compile(r'(Code|Référence|Type de contrat):\s*\d+', re.IGNORECASE),
            'list_item': re.compile(r'^\s*[-•*]\s+|\d+\.\s+', re.MULTILINE),
        }
    
    def extract_text_from_file(self, uploaded_file) -> Optional[Dict]:
        """Extract text with metadata from uploaded files"""
        filename = uploaded_file.name.lower()
        
        try:
            if filename.endswith('.pdf'):
                return self._extract_from_pdf(uploaded_file)
            elif filename.endswith('.txt'):
                text = uploaded_file.read().decode('utf-8')
                return {'text': text, 'pages': [{'page_num': 1, 'text': text}]}
            elif filename.endswith('.docx'):
                return self._extract_from_docx(uploaded_file)
            elif filename.endswith(('.csv', '.xlsx')):
                return self._extract_from_spreadsheet(uploaded_file, filename)
            else:
                return None
        except Exception as e:
            st.error(f"Error extracting from {uploaded_file.name}: {str(e)}")
            return None
    
    def _extract_from_pdf(self, uploaded_file) -> Dict:
        """Extract text from PDF with page-level granularity"""
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pages = []
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append({
                'page_num': page_num,
                'text': text
            })
        
        full_text = "\n".join([p['text'] for p in pages])
        return {'text': full_text, 'pages': pages}
    
    def _extract_from_docx(self, uploaded_file) -> Dict:
        """Extract text from DOCX"""
        doc = Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs])
        return {'text': text, 'pages': [{'page_num': 1, 'text': text}]}
    
    def _extract_from_spreadsheet(self, uploaded_file, filename: str) -> Dict:
        """Extract text from CSV/Excel"""
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        text = df.to_string(index=False)
        return {'text': text, 'pages': [{'page_num': 1, 'text': text}]}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n')
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+( of \d+)?', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def smart_chunk_document(self, doc_data: Dict, source_file: str) -> List[Dict]:
        """
        Intelligent chunking that preserves document structure
        Optimized for banking/technical documents
        """
        chunks = []
        pages = doc_data.get('pages', [])
        
        for page_info in pages:
            page_num = page_info['page_num']
            text = self.clean_text(page_info['text'])
            
            # Detect document structure
            if self._is_table_content(text):
                page_chunks = self._chunk_table_content(text, source_file, page_num)
            elif self._has_clear_sections(text):
                page_chunks = self._chunk_by_sections(text, source_file, page_num)
            else:
                page_chunks = self._chunk_semantic(text, source_file, page_num)
            
            chunks.extend(page_chunks)
        
        return chunks
    
    def _is_table_content(self, text: str) -> bool:
        """Detect if content is primarily tabular"""
        return bool(self.chunk_patterns['table_marker'].search(text))
    
    def _has_clear_sections(self, text: str) -> bool:
        """Detect if content has clear section structure"""
        headers = self.chunk_patterns['section_header'].findall(text)
        return len(headers) >= 2
    
    def _chunk_table_content(self, text: str, source: str, page: int) -> List[Dict]:
        """Chunk table content preserving structure"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            if current_size + line_size > Config.CHUNK_SIZES['table']:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk), source, page, 'table'
                    ))
                    # Overlap: keep last few lines
                    overlap_lines = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                    current_chunk = overlap_lines
                    current_size = sum(len(l) for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunks.append(self._create_chunk(
                '\n'.join(current_chunk), source, page, 'table'
            ))
        
        return chunks
    
    def _chunk_by_sections(self, text: str, source: str, page: int) -> List[Dict]:
        """Chunk by document sections"""
        chunks = []
        sections = re.split(r'(\n(?:Sect\.|Section|Chapitre|Article)\s+[\dIVX]+)', text, flags=re.IGNORECASE)
        
        current_section = ""
        
        for i, section in enumerate(sections):
            if i % 2 == 0:  # Content
                current_section += section
            else:  # Header
                if current_section.strip():
                    chunks.append(self._create_chunk(
                        current_section.strip(), source, page, 'section'
                    ))
                current_section = section
        
        if current_section.strip():
            chunks.append(self._create_chunk(
                current_section.strip(), source, page, 'section'
            ))
        
        return chunks
    
    def _chunk_semantic(self, text: str, source: str, page: int) -> List[Dict]:
        """Semantic chunking with overlap for regular content"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > Config.CHUNK_SIZES['paragraph']:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        ' '.join(current_chunk), source, page, 'paragraph'
                    ))
                    # Overlap: keep last 2 sentences
                    overlap = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                    current_chunk = overlap
                    current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(self._create_chunk(
                ' '.join(current_chunk), source, page, 'paragraph'
            ))
        
        return chunks
    
    def _create_chunk(self, text: str, source: str, page: int, chunk_type: str) -> Dict:
        """Create a standardized chunk object"""
        chunk_id = hashlib.md5(f"{source}_{page}_{text[:50]}".encode()).hexdigest()
        
        return {
            'id': chunk_id,
            'text': text,
            'metadata': {
                'source_file': source,
                'page': page,
                'chunk_type': chunk_type,
                'char_count': len(text),
                'timestamp': str(datetime.now())
            }
        }

# ==============================================================================
# BM25 IMPLEMENTATION WITH PERSISTENT INDEX
# ==============================================================================

class PersistentBM25:
    """BM25 implementation with disk persistence for large-scale retrieval"""
    
    def __init__(self, k1: float = Config.BM25_K1, b: float = Config.BM25_B):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_len = []
        self.doc_ids = []
        self.tokenized_corpus = []
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize with banking-specific preservation"""
        # Preserve codes and references
        text = re.sub(r'([A-Z]{2,}\d+)', r' \1 ', text)
        
        # Remove punctuation but keep hyphens in codes
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        
        # Tokenize
        tokens = text.split()
        
        # Filter stopwords (French banking context)
        stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'dans', 'pour', 'sur', 'avec'}
        tokens = [t for t in tokens if t and t not in stopwords]
        
        return tokens
    
    def fit(self, documents: List[Dict]):
        """Build BM25 index from documents"""
        self.corpus_size = len(documents)
        self.doc_ids = [doc['id'] for doc in documents]
        self.tokenized_corpus = []
        
        # Tokenize all documents
        for doc in documents:
            tokens = self.tokenize(doc['text'])
            self.tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            
            # Count document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate IDF values
        for token, freq in self.doc_freqs.items():
            self.idf[token] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
    
    def get_scores(self, query: str) -> Dict[str, float]:
        """Calculate BM25 scores for a query"""
        query_tokens = self.tokenize(query)
        scores = {}
        
        for idx, (doc_id, doc_tokens, doc_length) in enumerate(
            zip(self.doc_ids, self.tokenized_corpus, self.doc_len)
        ):
            score = 0
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                # Calculate term frequency in document
                tf = doc_tokens.count(token)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                score += self.idf[token] * (numerator / denominator)
            
            scores[doc_id] = score
        
        return scores
    
    def save(self, filepath: str):
        """Save BM25 index to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'k1': self.k1,
                'b': self.b,
                'corpus_size': self.corpus_size,
                'avgdl': self.avgdl,
                'doc_freqs': dict(self.doc_freqs),
                'idf': self.idf,
                'doc_len': self.doc_len,
                'doc_ids': self.doc_ids,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'PersistentBM25':
        """Load BM25 index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        bm25 = cls(k1=data['k1'], b=data['b'])
        bm25.corpus_size = data['corpus_size']
        bm25.avgdl = data['avgdl']
        bm25.doc_freqs = defaultdict(int, data['doc_freqs'])
        bm25.idf = data['idf']
        bm25.doc_len = data['doc_len']
        bm25.doc_ids = data['doc_ids']
        bm25.tokenized_corpus = data['tokenized_corpus']
        
        return bm25

# ==============================================================================
# QUERY EXPANSION FOR BANKING TERMINOLOGY
# ==============================================================================

class BankingQueryExpander:
    """Query expansion specialized for banking/financial terminology"""
    
    def __init__(self):
        # Banking-specific synonym mappings
        self.synonyms = {
            'compte': ['account', 'compte support', 'compte client'],
            'contrat': ['contract', 'agreement', 'convention'],
            'dépôt': ['deposit', 'versement', 'apport'],
            'garantie': ['guarantee', 'caution', 'warranty'],
            'échéance': ['maturity', 'deadline', 'due date', 'expiration'],
            'nantissement': ['pledge', 'collateral', 'guarantee'],
            'saisie': ['entry', 'input', 'capture'],
            'grille': ['grid', 'screen', 'form'],
            'transaction': ['operation', 'mouvement', 'transfer'],
            'client': ['customer', 'account holder', 'beneficiary'],
            'bénéficiaire': ['beneficiary', 'recipient', 'payee'],
            'taux': ['rate', 'interest rate', 'percentage'],
            'plafond': ['ceiling', 'limit', 'cap', 'maximum'],
            'retrait': ['withdrawal', 'extraction'],
            'virement': ['transfer', 'wire', 'payment'],
        }
        
        # Common banking abbreviations
        self.abbreviations = {
            'DAT': 'Dépôt à Terme',
            'CNT': 'Contrat',
            'GDI': 'Gestion De Interface',
            'BDC': 'Bon De Caisse',
            'SGE': 'Code siège',
            'DEV': 'Code devise',
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and banking terms"""
        expanded_queries = [query]
        
        query_lower = query.lower()
        
        # Add synonym expansions
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expanded_queries.append(expanded)
        
        # Expand abbreviations
        for abbr, full_form in self.abbreviations.items():
            if abbr in query:
                expanded = query.replace(abbr, full_form)
                expanded_queries.append(expanded)
            elif full_form.lower() in query_lower:
                expanded = query_lower.replace(full_form.lower(), abbr)
                expanded_queries.append(expanded)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:5]  # Limit to top 5 expansions

# ==============================================================================
# HYBRID RETRIEVAL SYSTEM
# ==============================================================================

class HybridRetriever:
    """Hybrid retrieval combining semantic search (ChromaDB) and keyword search (BM25)"""
    
    def __init__(self, chroma_collection, bm25_index: PersistentBM25, alpha: float = Config.HYBRID_ALPHA):
        self.chroma_collection = chroma_collection
        self.bm25_index = bm25_index
        self.alpha = alpha  # Weight between semantic (0) and keyword (1)
        self.query_expander = BankingQueryExpander()
    
    def retrieve(self, query: str, k: int = Config.INITIAL_RETRIEVAL_K) -> List[Dict]:
        """
        Hybrid retrieval with query expansion
        Returns top k documents based on weighted combination of semantic + keyword scores
        """
        # Expand query for better recall
        expanded_queries = self.query_expander.expand_query(query)
        
        all_results = {}
        
        for exp_query in expanded_queries:
            # 1. Semantic search via ChromaDB
            semantic_results = self.chroma_collection.query(
                query_texts=[exp_query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Normalize semantic scores (distances -> similarities)
            semantic_scores = {}
            if semantic_results['ids'] and semantic_results['ids'][0]:
                max_distance = max(semantic_results['distances'][0]) if semantic_results['distances'][0] else 1
                for idx, doc_id in enumerate(semantic_results['ids'][0]):
                    distance = semantic_results['distances'][0][idx]
                    similarity = 1 - (distance / max_distance) if max_distance > 0 else 1
                    semantic_scores[doc_id] = similarity
            
            # 2. Keyword search via BM25
            bm25_scores = self.bm25_index.get_scores(exp_query)
            
            # Normalize BM25 scores
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
            normalized_bm25 = {k: v / max_bm25 for k, v in bm25_scores.items()} if max_bm25 > 0 else bm25_scores
            
            # 3. Combine scores with weighted sum
            all_doc_ids = set(semantic_scores.keys()) | set(normalized_bm25.keys())
            
            for doc_id in all_doc_ids:
                sem_score = semantic_scores.get(doc_id, 0)
                bm25_score = normalized_bm25.get(doc_id, 0)
                
                # Hybrid score
                hybrid_score = (1 - self.alpha) * sem_score + self.alpha * bm25_score
                
                # Accumulate scores across expanded queries
                if doc_id in all_results:
                    all_results[doc_id]['score'] = max(all_results[doc_id]['score'], hybrid_score)
                else:
                    # Get document metadata
                    try:
                        doc_data = self.chroma_collection.get(
                            ids=[doc_id],
                            include=["documents", "metadatas"]
                        )
                        if doc_data['ids']:
                            all_results[doc_id] = {
                                'id': doc_id,
                                'text': doc_data['documents'][0],
                                'metadata': doc_data['metadatas'][0],
                                'score': hybrid_score,
                                'semantic_score': sem_score,
                                'bm25_score': bm25_score
                            }
                    except:
                        pass
        
        # Sort by score and return top k
        ranked_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
        return ranked_results[:k]

# ==============================================================================
# BATCH DOCUMENT INGESTION
# ==============================================================================

class BatchDocumentIngestion:
    """Optimized batch processing for large document collections"""
    
    def __init__(self, chroma_collection, processor: BankingDocumentProcessor):
        self.collection = chroma_collection
        self.processor = processor
        self.bm25_documents = []
    
    def ingest_documents(self, uploaded_files: List, progress_callback=None) -> Dict:
        """
        Batch ingest documents with progress tracking
        Returns statistics about the ingestion process
        """
        stats = {
            'total_files': len(uploaded_files),
            'processed_files': 0,
            'total_chunks': 0,
            'failed_files': [],
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        # Get existing documents to avoid duplicates
        existing_sources = self._get_existing_sources()
        
        batch_chunks = []
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            try:
                filename = uploaded_file.name
                
                # Skip if already indexed
                if filename in existing_sources:
                    if progress_callback:
                        progress_callback(file_idx + 1, len(uploaded_files), f"Skipped (already indexed): {filename}")
                    continue
                
                if progress_callback:
                    progress_callback(file_idx + 1, len(uploaded_files), f"Processing: {filename}")
                
                # Extract and chunk document
                doc_data = self.processor.extract_text_from_file(uploaded_file)
                
                if doc_data is None:
                    stats['failed_files'].append(filename)
                    continue
                
                chunks = self.processor.smart_chunk_document(doc_data, filename)
                
                # Add to batch
                batch_chunks.extend(chunks)
                self.bm25_documents.extend(chunks)
                
                # Process batch if it reaches batch size
                if len(batch_chunks) >= Config.BATCH_SIZE:
                    self._process_batch(batch_chunks)
                    stats['total_chunks'] += len(batch_chunks)
                    batch_chunks = []
                
                stats['processed_files'] += 1
                
            except Exception as e:
                stats['failed_files'].append(f"{uploaded_file.name}: {str(e)}")
        
        # Process remaining chunks
        if batch_chunks:
            self._process_batch(batch_chunks)
            stats['total_chunks'] += len(batch_chunks)
        
        stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return stats
    
    def _process_batch(self, chunks: List[Dict]):
        """Process a batch of chunks into ChromaDB"""
        if not chunks:
            return
        
        ids = [chunk['id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    def _get_existing_sources(self) -> set:
        """Get list of already indexed source files"""
        try:
            all_data = self.collection.get(include=["metadatas"])
            if all_data and all_data['metadatas']:
                return set(meta.get('source_file', '') for meta in all_data['metadatas'])
        except:
            pass
        return set()
    
    def build_bm25_index(self) -> PersistentBM25:
        """Build and save BM25 index from ingested documents"""
        bm25 = PersistentBM25()
        bm25.fit(self.bm25_documents)
        bm25.save(Config.BM25_INDEX_PATH)
        return bm25

# ==============================================================================
# RAG GENERATION WITH IMPROVED CONTEXT
# ==============================================================================

class BankingRAGGenerator:
    """Enhanced RAG generation for banking documentation"""
    
    def __init__(self, azure_client):
        self.client = azure_client
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate response with improved context assembly"""
        
        # Assemble context with source citations
        context_parts = []
        sources = []
        
        for idx, doc in enumerate(context_docs, 1):
            metadata = doc.get('metadata', {})
            source_file = metadata.get('source_file', 'Unknown')
            page = metadata.get('page', 'N/A')
            
            context_parts.append(
                f"[Document {idx} - {source_file}, Page {page}]\n{doc['text']}\n"
            )
            
            if source_file not in sources:
                sources.append(source_file)
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt for banking documentation
        prompt = f"""Vous êtes un assistant RAG spécialisé dans les systèmes bancaires, l'architecture de données, et les produits financiers.

Votre rôle est de répondre et d'expliquer clairement les informations issues des documents, en les rendant compréhensibles pour tout type d'utilisateur, du plus général au plus technique.

[Contexte]
{context}

[Requête]
{query}

[INSTRUCTIONS GÉNÉRALES]

1. Utilisez exclusivement les informations présentes dans le contexte ci-dessus.

2. Fournissez une réponse claire, structurée et explicite, adaptée à la compréhension de tout utilisateur.
   - Si le sujet est technique, expliquez les notions en termes simples.
   - Si la question est générale, donnez une réponse complète mais concise.
   - Si la question est complexe, détaillez le raisonnement et le fonctionnement.

3. Chaque fois que vous mentionnez une donnée, un mécanisme ou un fait:
   - Citez immédiatement la source sous le format: [Source: nom_du_fichier.pdf, Page <numéro>]

4. Si l'information n'existe pas dans les documents, indiquez-le clairement.

5. Structure attendue:
   a. Réponse expliquée: Détaillez le contenu et son interprétation.
   b. Synthèse (si utile): Résumez la logique ou le fonctionnement global.
   c. Sources: Liste complète des documents utilisés.

6. Objectif:
   Rendre la réponse à la fois informative, explicative et vérifiable,
   qu'il s'agisse d'un utilisateur curieux ou d'un expert technique.

[Exemple de format]
Les procédures de vérification d'identité... [Source: procedure_kyc.pdf, Page 12]. 
Selon le chapitre 3... [Source: reglement_financier.pdf, Page 45].

Sources utilisées:
- procedure_kyc.pdf (Pages 12, 15)
- reglement_financier.pdf (Page 45)
"""
        
        try:
            completion = self.client.chat.completions.create(
                model=Config.AZURE_MODEL_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert assistant for banking system documentation based on RAG (Retrieval-Augmented Generation)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            answer = completion.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': sources,
                'context_docs': context_docs,
                'num_tokens': completion.usage.total_tokens if hasattr(completion, 'usage') else None
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'context_docs': [],
                'num_tokens': None
            }

# ==============================================================================
# STREAMLIT APPLICATION
# ==============================================================================

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_PATH)
    
    if 'collection' not in st.session_state:
        # Load embedding model
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH)
        
        # Create embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL_PATH
        )
        
        st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    if 'bm25_index' not in st.session_state:
        # Load or create BM25 index
        if os.path.exists(Config.BM25_INDEX_PATH):
            st.session_state.bm25_index = PersistentBM25.load(Config.BM25_INDEX_PATH)
        else:
            st.session_state.bm25_index = PersistentBM25()
    
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = str(datetime.now())
        st.session_state.chat_sessions[st.session_state.current_chat] = {
            "title": "New Chat",
            "messages": []
        }
    
    if 'processor' not in st.session_state:
        st.session_state.processor = BankingDocumentProcessor()
    
    if 'azure_client' not in st.session_state:
        # Initialize Azure OpenAI client
        http_client = httpx.Client(verify=False)
        st.session_state.azure_client = AzureOpenAI(
            api_version=Config.AZURE_API_VERSION,
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            http_client=http_client
        )
    
    if 'rag_generator' not in st.session_state:
        st.session_state.rag_generator = BankingRAGGenerator(st.session_state.azure_client)

def main():
    st.set_page_config(page_title="BNP Banking RAG Chatbot", layout="wide")
    
    initialize_session_state()
    
    st.title("🏦 BNP Banking Documentation Chatbot")
    st.caption("Optimized RAG system with hybrid search and batch processing")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Get collection stats
        try:
            collection_count = st.session_state.collection.count()
            st.metric("Indexed Documents", collection_count)
        except:
            collection_count = 0
            st.metric("Indexed Documents", "0")
        
        # Batch upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload multiple documents",
            type=["pdf", "txt", "docx", "csv", "xlsx"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📤 Process All Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(message)
            
            with st.spinner("Processing documents..."):
                batch_ingestion = BatchDocumentIngestion(
                    st.session_state.collection,
                    st.session_state.processor
                )
                
                stats = batch_ingestion.ingest_documents(
                    uploaded_files,
                    progress_callback=update_progress
                )
                
                # Rebuild BM25 index
                status_text.text("Building BM25 index...")
                st.session_state.bm25_index = batch_ingestion.build_bm25_index()
                
                # Display results
                st.success(f"""
                ✅ Processing complete!
                - Files processed: {stats['processed_files']}/{stats['total_files']}
                - Total chunks: {stats['total_chunks']}
                - Time: {stats['processing_time']:.2f}s
                """)
                
                if stats['failed_files']:
                    st.error(f"Failed files: {', '.join(stats['failed_files'])}")
        
        st.divider()
        
        # Chat management
        st.subheader("💬 Chat Sessions")
        if st.button("➕ New Chat"):
            new_chat_id = str(datetime.now())
            st.session_state.current_chat = new_chat_id
            st.session_state.chat_sessions[new_chat_id] = {
                "title": "New Chat",
                "messages": []
            }
            st.rerun()
        
        # List existing chats
        for chat_id, chat_data in list(st.session_state.chat_sessions.items()):
            if st.button(f"📝 {chat_data['title'][:30]}", key=f"chat_{chat_id}"):
                st.session_state.current_chat = chat_id
                st.rerun()
    
    # Main chat interface
    current_chat = st.session_state.chat_sessions.get(st.session_state.current_chat, {
        "title": "New Chat",
        "messages": []
    })
    
    # Display chat messages
    for message in current_chat['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            # Show sources if available
            if message['role'] == 'assistant' and 'sources' in message:
                with st.expander("📚 Sources"):
                    for source in message['sources']:
                        st.text(f"• {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask about banking documentation..."):
        # Add user message
        current_chat['messages'].append({
            "role": "user",
            "content": prompt,
            "timestamp": str(datetime.now())
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documentation..."):
                # Create hybrid retriever
                retriever = HybridRetriever(
                    st.session_state.collection,
                    st.session_state.bm25_index
                )
                
                # Retrieve relevant documents
                retrieved_docs = retriever.retrieve(prompt, k=Config.FINAL_RESULTS_K)
                
                # Generate response
                response = st.session_state.rag_generator.generate_response(
                    prompt,
                    retrieved_docs
                )
                
                # Display answer
                st.markdown(response['answer'])
                
                # Show sources
                if response['sources']:
                    with st.expander("📚 Sources"):
                        for source in response['sources']:
                            st.text(f"• {source}")
                
                # Add to chat history
                current_chat['messages'].append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response['sources'],
                    "timestamp": str(datetime.now())
                })
                
                # Update chat title if first message
                if current_chat['title'] == "New Chat" and len(current_chat['messages']) > 0:
                    current_chat['title'] = prompt[:50] + "..."

if __name__ == "__main__":
    main()
