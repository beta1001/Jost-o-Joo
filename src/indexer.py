# -*- coding: utf-8 -*-
"""
Search Indexer Module for Jost-o-Joo Search Engine

This module implements TF-IDF indexing and inverted index construction.
It processes document collections, calculates term weights, and builds
searchable indexes for the vector space model.

Key Features:
- TF-IDF weight calculation
- Inverted index construction
- Text preprocessing (tokenization, stemming, stopword removal)
- Document vector and norm computation


Created on Sat Dec 13 19:58:39 2025

@author: Rihem Touzi
"""

import json
import re
import math
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent.parent

class SearchIndexer:
    """
    Builds and manages the search index using TF-IDF and inverted indexing.
    
    The SearchIndexer creates an inverted index where each term points to
    documents containing it, with associated TF-IDF weights. It also computes
    document vectors and their norms for efficient cosine similarity calculations.
    
    Attributes:
        metadata_file (Path): Path to metadata JSON file
        index_file (Path): Path to save inverted index
        metadata (dict): Loaded document metadata
        inverted_index (defaultdict): term -> {doc_id: tf_idf_score}
        doc_vectors (dict): doc_id -> {term: tf_idf_score}
        doc_norms (dict): doc_id -> vector norm for cosine similarity
        stemmer (PorterStemmer): For stemming terms
        stop_words (set): English stopwords to filter out
    """
    def __init__(self,
             metadata_file=BASE_DIR / "data" / "metadata.json",
             index_file=BASE_DIR / "index" / "inverted_index.json"):
        # Use relative paths from src/ directory
        self.metadata_file = Path(metadata_file)
        self.index_file = Path(index_file)
        self.metadata = {}
        self.inverted_index = defaultdict(dict)  # term -> {doc_id: tf_idf_score}
        self.doc_vectors = {}  # doc_id -> {term: tf_idf_score}
        self.doc_norms = {}  # doc_id -> norm of vector
        self.stemmer = PorterStemmer()
        
        # Ensure index directory exists
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from JSON file"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f" Loaded metadata for {len(self.metadata)} documents")
        except FileNotFoundError:
            print(f"❌ Metadata file not found: {self.metadata_file.absolute()}")
            print("   Run data_collector.py first!")
    
    def preprocess_text(self, text):
        """
        Preprocess text: tokenize, remove stopwords, and stem.
        
        Steps:
        1. Convert to lowercase
        2. Remove non-alphabetic characters
        3. Tokenize by whitespace
        4. Filter stopwords and short terms (<3 chars)
        5. Apply Porter stemming
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            list: Preprocessed tokens
        """
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Remove special characters and numbers, keep only letters
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # 3.Tokenize
        tokens = text.split()
        
        # 4. Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # 5. Apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def build_inverted_index(self):
        """
        Build inverted index with TF-IDF weights.
        
        Process:
        1. For each document:
           - Read content
           - Preprocess to tokens
           - Calculate term frequencies
        2. Calculate IDF for each term
        3. Compute TF-IDF weights
        4. Build document vectors and norms
        
        Returns:
            bool: True if successful, False otherwise
        """
        
        if not self.metadata:
            print("❌ No metadata found. Run data_collector.py first!")
            return False
        
        print(" Building inverted index...")
        
        # Document frequency (DF) for each term
        doc_freq = defaultdict(int)
        
        # Process each document
        for doc_id, meta in self.metadata.items():
            content = self._get_document_content(doc_id)
            if not content:
                continue
            
            # Preprocess and tokenize
            tokens = self.preprocess_text(content)
            
            # Calculate term frequency (TF)
            tf_counter = Counter(tokens)
            total_terms = len(tokens)
            
            # Store TF in inverted index
            for term, count in tf_counter.items():
                tf = count / total_terms  # Normalized TF
                self.inverted_index[term][doc_id] = tf
                doc_freq[term] += 1
            
            print(f"  ✓ Processed {doc_id}: {len(tokens)} tokens")
        
        # Calculate TF-IDF weights
        total_docs = len(self.metadata)
        print(f"\n Calculating TF-IDF weights for {len(self.inverted_index)} terms...")
        
        for term, doc_dict in self.inverted_index.items():
            idf = math.log(total_docs / (1 + doc_freq[term]))  # Add 1 to avoid division by zero
            
            for doc_id, tf in doc_dict.items():
                tf_idf = tf * idf
                self.inverted_index[term][doc_id] = tf_idf
                
                # Build document vectors
                if doc_id not in self.doc_vectors:
                    self.doc_vectors[doc_id] = {}
                self.doc_vectors[doc_id][term] = tf_idf
        
        # Calculate document norms for cosine similarity
        for doc_id, vector in self.doc_vectors.items():
            norm = math.sqrt(sum(score ** 2 for score in vector.values()))
            self.doc_norms[doc_id] = norm if norm > 0 else 1
        
        print(f"✅ Inverted index built with {len(self.inverted_index)} unique terms")
        return True
    
    def _get_document_content(self, doc_id):
        """
        Get content of a specific document.
        
        Args:
            doc_id (str): Document identifier
            
        Returns:
            str or None: Document content if found, None otherwise
        """
        if doc_id in self.metadata:
            doc_path = Path(self.metadata[doc_id]['path'])
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"  ✗ Error reading {doc_id}: {e}")
        return None
    
    def save_index(self):
        """
        Save inverted index to JSON and pickle files.
        
        Saves two formats:
        1. JSON: Human-readable, for inspection
        2. Pickle: Binary format, faster loading
        
        Creates files:
            - index/inverted_index.json
            - index/inverted_index.pkl
        """
        # Convert defaultdict to regular dict for JSON serialization
        index_dict = {term: dict(doc_dict) for term, doc_dict in self.inverted_index.items()}
        
        index_data = {
            'inverted_index': index_dict,
            'doc_vectors': {doc_id: dict(vector) for doc_id, vector in self.doc_vectors.items()},
            'doc_norms': dict(self.doc_norms),
            'metadata_summary': {
                'total_docs': len(self.metadata),
                'total_terms': len(self.inverted_index)
            }
        }
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, default=str)
        
        print(f" Index saved to: {self.index_file.absolute()}")
        
        # Also save as pickle for faster loading
        pickle_file = self.index_file.with_suffix('.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'inverted_index': self.inverted_index,
                'doc_vectors': self.doc_vectors,
                'doc_norms': self.doc_norms
            }, f)
        
        print(f" Pickle version saved to: {pickle_file.absolute()}")
    
    def load_index(self):
        """
        Load inverted index from JSON file.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Convert back to defaultdict
            self.inverted_index = defaultdict(dict)
            for term, doc_dict in index_data['inverted_index'].items():
                self.inverted_index[term] = doc_dict
            
            self.doc_vectors = {doc_id: dict(vector) for doc_id, vector in index_data['doc_vectors'].items()}
            self.doc_norms = index_data['doc_norms']
            
            print(f"📂 Loaded index with {len(self.inverted_index)} terms")
            return True
        except FileNotFoundError:
            print(f"❌ Index file not found: {self.index_file.absolute()}")
            return False
    
    def get_term_info(self, term):
        """ 
        Get information about a specific term
        Get information about a specific term in the index.
        
        Args:
            term (str): Term to look up
            
        Returns:
            dict or None: Term information including:
                - term: Stemmed term
                - document_frequency: Number of documents containing term
                - documents: List of document IDs
                - scores: TF-IDF scores per document
        """
        stemmed_term = self.stemmer.stem(term.lower())
        if stemmed_term in self.inverted_index:
            doc_dict = self.inverted_index[stemmed_term]
            return {
                'term': stemmed_term,
                'document_frequency': len(doc_dict),
                'documents': list(doc_dict.keys()),
                'scores': doc_dict
            }
        return None
    
    def get_document_vector(self, doc_id):
        """
        Get TF-IDF vector for a document
        Args:
            doc_id (str): Document identifier
            
        Returns:
            dict: Term -> TF-IDF score mapping
        """
        return self.doc_vectors.get(doc_id, {})
    
    def get_index_stats(self):
        """
        Get statistics about the index.
        
        Returns:
            dict: Index statistics including:
                - total_documents: Number of documents indexed
                - total_terms: Number of unique terms
                - avg_terms_per_doc: Average terms per document
        """
        return {
            'total_documents': len(self.metadata),
            'total_terms': len(self.inverted_index),
            'avg_terms_per_doc': sum(len(v) for v in self.doc_vectors.values()) / len(self.doc_vectors) if self.doc_vectors else 0
        }

def main():
    """Main function to run indexing"""
    print("=" * 60)
    print("🔍 SEARCH INDEXER MODULE")
    print("=" * 60)
    
    indexer = SearchIndexer()
    
    if indexer.build_inverted_index():
        indexer.save_index()
        
        stats = indexer.get_index_stats()
        print("\n Index Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Total Unique Terms: {stats['total_terms']}")
        print(f"  Average Terms per Document: {stats['avg_terms_per_doc']:.2f}")
        
        # Show sample terms
        print("\n Sample Terms in Index:")
        sample_terms = list(indexer.inverted_index.keys())[:10]
        for term in sample_terms:
            info = indexer.get_term_info(term)
            if info:
                print(f"  '{term}': appears in {info['document_frequency']} documents")
    else:
        print("❌ Index building failed!")

if __name__ == "__main__":
    main()