# Jost-o-Joo Search Engine 🔍

A vector space model search engine built in Python for searching Project Gutenberg books.

## Features

- **Vector Space Model**: TF-IDF with cosine similarity
- **Full-text search**: Fast document retrieval
- **Evaluation metrics**: Precision, Recall, F1-Score, MAP
- **Rich + Click CLI interface**: Beautiful terminal output
- **Modular architecture**: Easy to extend and modify

## Installation

### 1. Prerequisites

```
Python 3.8+
Git
```

### 2. Clone Repository

```
git clone https://github.com/yourusername/Jost-o-Joo.git
cd Jost-o-Joo
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Download NLTK Data

```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### Step 1: Download Documents

```
python data/downloadTxtFiles.py
```

### Step 2: Collect Metadata

```
python src/main.py collect
```

then run fix_titles.py

```
python src/fix_titles.py
```

### Step 3: Build Index

```
python src/main.py index
```

### Step 4: Search Documents

```
python src/main.py search "love and romance"
```

### Step 5: generate ground_truh.csv file

```
python src/generate_ground_truth.py
```

### Step 6: Evaluate Performance

```
python src/main.py evaluate
```

## Commands Overview


Command	     | Description	            | Options
----------------------------------------------------------
collect	     |Collect document metadata	| --dir
index	     | Build search index	    | --metadata, --index
search	     | Search for documents	    | --top, --explain
evaluate	 | Evaluate search engine	| --k, --queries-file
stats	     | Show statistics	        |
test_queries |	Show test queries       |
## Examples
### Basic Search
```
python src/main.py search "adventure journey" --top 5
```
### Search with Explanation
```
python src/main.py search "detective mystery" --explain
```
### Detailed Evaluation
```
python src/main.py evaluate --k 5 --queries-file tests/test_queries.txt
```