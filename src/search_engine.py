# -*- coding: utf-8 -*-
"""
Vector Space Search Engine Module for Jost-o-Joo Search Engine

This module implements the core search functionality using the Vector Space Model.
It performs query processing, TF-IDF vector calculation, cosine similarity scoring,
and result retrieval.

Key Features:
- Query preprocessing (tokenization, stemming, stopword removal)
- TF-IDF vector construction for queries
- Cosine similarity calculation
- Document snippet generation
- Search explanation and debugging

Created on Sat Dec 13 19:59:13 2025

@author: Rihem Touzi
"""

import json
import math
import re
import time
from pathlib import Path
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# =======================
# Project root directory
# =======================
BASE_DIR = Path(__file__).resolve().parent.parent


class VectorSpaceSearchEngine:
    """
    Implements a search engine using the Vector Space Model with TF-IDF.
    
    This class provides the core search functionality including:
    - Query processing and vectorization
    - Cosine similarity computation
    - Document ranking and retrieval
    - Snippet generation for search results
    
    Attributes:
        index_file (Path): Path to inverted index JSON file
        metadata_file (Path): Path to metadata JSON file
        inverted_index (defaultdict): term -> {doc_id: tf_idf_score}
        doc_vectors (dict): doc_id -> {term: tf_idf_score}
        doc_norms (dict): doc_id -> vector norm for cosine similarity
        metadata (dict): Document metadata (titles, authors, etc.)
        stemmer (PorterStemmer): Stemmer for text normalization
        stop_words (set): English stopwords to filter
    """
    def __init__(
        self,
        index_file=BASE_DIR / "index" / "inverted_index.json",
        metadata_file=BASE_DIR / "data" / "metadata.json",
    ):
        """
       Initialize the VectorSpaceSearchEngine.
       
       Args:
           index_file (Path/str): Path to inverted index file
           metadata_file (Path/str): Path to metadata file
           
       Loads:
           - Inverted index (pre-built by SearchIndexer)
           - Document vectors and norms
           - Document metadata
           - NLTK stopwords (downloads if missing)
           
       Raises:
           FileNotFoundError: If index or metadata files don't exist
       """
        self.index_file = Path(index_file)
        self.metadata_file = Path(metadata_file)

        self.inverted_index = defaultdict(dict)
        self.doc_vectors = {}
        self.doc_norms = {}
        self.metadata = {}

        self.stemmer = PorterStemmer()
        self.stop_words = self._load_stopwords()

        # Load resources
        self._load_index()
        self._load_metadata()

    # =======================
    # Resource loading
    # =======================
    def _load_stopwords(self):
        """
        Safely load NLTK stopwords, downloading if necessary.
        
        Returns:
            set: English stopwords
        """
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            return set(stopwords.words("english"))

    def _load_index(self):
        """
       Load inverted index from JSON file.
       
       The index file must contain:
           - inverted_index: term -> {doc_id: tf_idf_score}
           - doc_vectors: doc_id -> {term: tf_idf_score}
           - doc_norms: doc_id -> vector norm
           
       Raises:
           FileNotFoundError: If index file doesn't exist
       """
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"❌ Index file not found: {self.index_file}\n"
                f"   Run: python src/main.py index"
            )

        with open(self.index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.inverted_index.update(index_data["inverted_index"])
        self.doc_vectors = index_data["doc_vectors"]
        self.doc_norms = index_data["doc_norms"]

        print(f"🔍 Loaded index with {len(self.inverted_index)} terms")

    def _load_metadata(self):
        """
        Load metadata from JSON file.
        
        Metadata includes:
            - Document titles
            - Authors
            - Word counts
            - File paths
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"❌ Metadata file not found: {self.metadata_file}\n"
                f"   Run: python src/main.py collect"
            )

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    # =======================
    # Query processing
    # =======================
    def preprocess_query(self, query):
        """
        Preprocess a search query using the same pipeline as document indexing.
        
        Steps:
        1. Convert to lowercase
        2. Remove non-alphabetic characters
        3. Tokenize by whitespace
        4. Filter stopwords and short terms (<3 chars)
        5. Apply Porter stemming
        
        Args:
            query (str): Raw user query
            
        Returns:
            list: Preprocessed query terms
        """
        # 1. Normalize to lowercase
        query = query.lower()
        # 2. Remove special characters, keep only letters and spaces
        query = re.sub(r"[^a-z\s]", " ", query)
        # 3. Tokenize and process
        tokens = query.split()
        # 4. Filter stopwords and short terms 
        tokens = [
            #5. Apply Porter stemming
            self.stemmer.stem(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return tokens

    def calculate_query_vector(self, query_terms):
        """
        Calculate TF-IDF vector for a processed query.
        
        Process:
        1. Count term frequencies in query
        2. Calculate TF: term_frequency / total_terms
        3. Get IDF from inverted index
        4. Compute TF-IDF: TF × IDF
        
        Args:
            query_terms (list): Preprocessed query terms
            
        Returns:
            dict: Query vector {term: tf_idf_score}
        """
        # Count term frequencies in query
        tf_counter = defaultdict(int)
        for term in query_terms:
            tf_counter[term] += 1

        total_terms = len(query_terms)
        query_vector = {}
        # Calculate TF-IDF for each term
        for term, count in tf_counter.items():
            if term in self.inverted_index:
                tf = count / total_terms   # Term frequency in query
                idf = self._calculate_idf(term)    # Inverse document frequency
                query_vector[term] = tf * idf

        return query_vector

    def _calculate_idf(self, term):
        """
        Calculate Inverse Document Frequency (IDF) for a term.
        
        Formula: IDF(t) = log(N / (1 + df(t)))
        Where:
            N = total documents
            df(t) = document frequency (documents containing term)
            
        Args:
            term (str): Term to calculate IDF for
            
        Returns:
            float: IDF score
        """
        total_docs = len(self.doc_vectors)
        doc_freq = len(self.inverted_index.get(term, {}))
        return math.log(total_docs / (1 + doc_freq)) if doc_freq else 0

    # =======================
    # Similarity
    # =======================
    def cosine_similarity(self, query_vector, doc_id):
        """
        Calculate cosine similarity between query and document.
        
        Formula: cos(θ) = (A·B) / (||A|| * ||B||)
        Where:
            A·B = dot product of query and document vectors
            ||A|| = norm of query vector
            ||B|| = norm of document vector (precomputed)
            
        Args:
            query_vector (dict): Query TF-IDF vector
            doc_id (str): Document identifier
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        # Get document vector
        doc_vector = self.doc_vectors.get(doc_id)
        if not doc_vector:
            return 0
        # Calculate dot product: Σ query_term_score × doc_term_score
        dot = sum(
            query_vector[t] * doc_vector.get(t, 0)
            for t in query_vector
        )
        # Calculate query vector norm

        query_norm = math.sqrt(sum(v ** 2 for v in query_vector.values()))
        # Get precomputed document norm
        doc_norm = self.doc_norms.get(doc_id, 1)
        # Avoid division by zero
        if query_norm == 0 or doc_norm == 0:
            return 0
        # Cosine similarity
        return dot / (query_norm * doc_norm)

    # =======================
    # Search
    # =======================
    def search(self, query, top_k=10):
        """
        Execute a search query and return ranked results.
        
        Process:
        1. Preprocess query
        2. Build query vector
        3. Gather candidate documents from inverted index
        4. Calculate cosine similarity for each candidate
        5. Sort by score and return top-k results
        
        Args:
            query (str): User search query
            top_k (int): Number of results to return
            
        Returns:
            tuple: (results, search_time, message) where:
                - results: List of dicts with doc_id, score, metadata
                - search_time: Time taken for search in seconds
                - message: Status message
        """
        start_time = time.time()
        # Step 1: Preprocess query
        query_terms = self.preprocess_query(query)
        if not query_terms:
            return [], 0, "No valid search terms"
        # Step 2: Calculate query vector
        query_vector = self.calculate_query_vector(query_terms)
        if not query_vector:
            return [], 0, "No matching terms in index"
        # Step 3: Gather candidate documents
        candidate_docs = set()
        for term in query_vector:
            candidate_docs.update(self.inverted_index[term].keys())
        # Step 4: Score each candidate
        scores = []
        for doc_id in candidate_docs:
            score = self.cosine_similarity(query_vector, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        # Step 5: Sort and select top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]
        # Format results with metadata
        results = [
            {
                "doc_id": doc_id,
                "score": score,
                "metadata": self.metadata.get(doc_id, {}),
            }
            for doc_id, score in top_results
        ]

        return results, time.time() - start_time, f"Found {len(scores)} documents"

    # =======================
    # Snippets
    # =======================
    def get_document_snippet(self, doc_id, query_terms, max_length=200):
        """
        Generate a contextual snippet for a search result.
        
        Tries to find query terms in document and extract surrounding text.
        If no query terms found, returns beginning of document.
        
        Args:
            doc_id (str): Document identifier
            query_terms (list): Preprocessed query terms
            max_length (int): Maximum snippet length
            
        Returns:
            str: Document snippet with query terms highlighted
        """
        
        meta = self.metadata.get(doc_id)
        if not meta or "path" not in meta:
            return "No content available"
        # Resolve document path
        doc_path = Path(meta["path"])
        if not doc_path.is_absolute():
            doc_path = BASE_DIR / doc_path

        try:
            content = doc_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"Error reading document: {e}"
        # Try to find query terms for context-aware snippet
        content_lower = content.lower()
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1:
                # Extract text around the term
                start = max(0, pos - 50)
                end = min(len(content), pos + 150)
                snippet = content[start:end]
                return f"...{snippet}..."
        # Fallback: beginning of document
        return content[:max_length] + ("..." if len(content) > max_length else "")

    # =======================
    # Debug / Explain
    # =======================
    def explain_search(self, query):
        """
        Generate debug information about a search query.
        
        Useful for understanding why certain results are returned.
        
        Args:
            query (str): Search query
            
        Returns:
            dict: Explanation containing:
                - original_query: Raw input query
                - processed_terms: After preprocessing
                - query_vector: TF-IDF scores for terms
                - matching_terms: Terms found in index
        """
        terms = self.preprocess_query(query)
        vector = self.calculate_query_vector(terms)

        return {
            "original_query": query,
            "processed_terms": terms,
            "query_vector": {k: round(v, 4) for k, v in vector.items()},
            "matching_terms": [t for t in terms if t in self.inverted_index],
        }


if __name__ == "__main__":
    engine = VectorSpaceSearchEngine()
    results, t, msg = engine.search("test query")
    print(msg)
