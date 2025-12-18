# Architecture Documentation 

## Overview
Jost-o-Joo is a modular search engine implementing the Vector Space Model with TF-IDF weighting and cosine similarity.

## System Architecture

```
graph TB
    A[User Query] --> B[CLI Interface]
    B --> C[Search Engine]
    C --> D[Query Processing]
    D --> E[TF-IDF Vector]
    E --> F[Cosine Similarity]
    F --> G[Ranked Results]
    
    H[Documents] --> I[Data Collector]
    I --> J[Metadata]
    J --> K[Indexer]
    K --> L[Inverted Index]
    L --> C
    
    M[Test Queries] --> N[Evaluator]
    N --> O[Metrics]
    O --> P[Evaluation Report]
```
## Project Structure
Jost-o-Joo/
├── data/
│   ├── documents/          # Text files
│   ├── metadata.json       # Document metadata
│   └── downloadTxtFiles.py # Document downloader
├── src/
│   ├── main.py            # CLI interface
│   ├── data_collector.py  # Metadata collection
│   ├── indexer.py         # TF-IDF indexing
│   ├── search_engine.py   # Search algorithms
│   ├── evaluator.py       # Evaluation metrics
│   ├── generate_ground_truth .py   # generates ground_truth.csv
│   └── fix_titles.py      # fixes titels of books after running data_collector.py
├── index/
│   ├── inverted_index.json
│   └── inverted_index.pkl
├── tests/
│   └── test_queries.txt
├── ground_truth.csv       # Relevance judgments
├── requirements.txt       # Dependencies
├── README.md             #  README file
└── ARCHITECTURE.md       # Technical documentation (This file)*
## Modules Description
### 1. Main Module (src/main.py)
Purpose: CLI interface and coordination
Components:
CLI commands using Click
Progress tracking with Rich
Module orchestration
Dependencies: All other modules

### 2. Data Collector (src/data_collector.py)
Purpose: Document metadata extraction
Process:
Scan document directory
Extract titles/authors from content
Generate preview snippets
Save metadata to JSON
Algorithms:
Pattern matching for Gutenberg headers
Title extraction heuristics
Output: data/metadata.json

### 3. Indexer (src/indexer.py)
Purpose: Build search index using TF-IDF

Algorithms:

```
# TF-IDF Calculation
TF = term_frequency / document_length
IDF = log(total_documents / document_frequency)
TF-IDF = TF × IDF
```
Process:

Text preprocessing (tokenization, stemming, stopwords)

Term frequency counting

TF-IDF weight calculation

Document vector normalization

Output: index/inverted_index.json

### 4. Search Engine (src/search_engine.py)
Purpose: Query processing and document ranking

Algorithms:

```
# Cosine Similarity
similarity = dot(query_vector, doc_vector) / (norm(query) × norm(doc))
```
-Process:

Query preprocessing (same as indexing)

Query vector creation

Cosine similarity calculation

Result ranking

-Features:

Top-k results retrieval

Snippet generation

Search explanation

### 5. Evaluator (src/evaluator.py)
Purpose: Search engine performance evaluation

-Metrics:

Precision@k

Recall@k

F1-Score

Average Precision (AP)

Mean Average Precision (MAP)

nDCG

-Process:

Load test queries and ground truth

Execute searches

Calculate metrics

Generate reports
## Data Flow
### Indexing Pipeline
Documents → Data Collector → Metadata →fix_titles → Indexer → Inverted Index
### Search Pipeline
User Query → Search Engine → Preprocessing → Vectorization → 
Similarity Calculation → Ranking → Results
### Evaluation Pipeline
Test Queries → Evaluator → Search Engine → Results → 
Metrics Calculation → Report Generation
##Technology Stack
Language: Python 3.8+

NLP: NLTK for text processing

CLI: Click for command interface

UI: Rich for terminal formatting

Data: JSON for storage

Math: NumPy for vector operations
## Core Dependencies
click==8.1.7
rich==14.2.0
nltk==3.8.1
scikit-learn==1.5.0
requests==2.31.0
beautifulsoup4==4.12.2