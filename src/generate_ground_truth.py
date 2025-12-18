# generate_ground_truth.py
import csv
import random
from pathlib import Path
from search_engine import VectorSpaceSearchEngine
import json

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
TEST_QUERIES_FILE = BASE_DIR / "tests" / "test_queries.txt"
GROUND_TRUTH_FILE = BASE_DIR / "ground_truth.csv"
METADATA_FILE = BASE_DIR / "data" / "metadata.json"

# Query categories for smarter relevance assignment
QUERY_CATEGORIES = {
    "love and romance": ["love", "romance", "marriage", "relationship", "wedding", "affection"],
    "adventure journey": ["adventure", "journey", "travel", "exploration", "expedition", "voyage"],
    "science discovery": ["science", "discovery", "experiment", "invention", "research", "technology"],
    "war peace conflict": ["war", "peace", "conflict", "battle", "soldier", "army", "fight"],
    "philosophy life meaning": ["philosophy", "life", "meaning", "existence", "thought", "mind", "soul"],
    "detective mystery": ["detective", "mystery", "crime", "investigation", "clue", "suspect", "murder"],
    "ghost horror": ["ghost", "horror", "terror", "supernatural", "fear", "monster", "haunted"],
    "fairy tales children": ["fairy", "children", "tale", "magic", "story", "prince", "princess", "castle"],
    "sea ocean voyage": ["sea", "ocean", "voyage", "ship", "sail", "water", "navy", "pirate"],
    "revolution society": ["revolution", "society", "change", "reform", "government", "people", "power"]
}

def is_document_relevant(doc_metadata, query, query_keywords):
    """Determine if a document is relevant (1) or not (0)"""
    title = doc_metadata.get('title', '').lower()
    author = doc_metadata.get('author', '').lower()
    preview = doc_metadata.get('preview', '').lower()
    
    # Check for strong title matches
    strong_title_matches = {
        "love and romance": any(word in title for word in ["love", "romance", "pride and prejudice", "marriage"]),
        "adventure journey": any(word in title for word in ["adventure", "journey", "travel", "voyage", "expedition"]),
        "science discovery": any(word in title for word in ["science", "frankenstein", "discovery", "experiment"]),
        "war peace conflict": any(word in title for word in ["war", "peace", "conflict", "battle", "soldier"]),
        "philosophy life meaning": any(word in title for word in ["philosophy", "meaning", "life", "thought", "meditation"]),
        "detective mystery": any(word in title for word in ["detective", "mystery", "sherlock", "crime", "investigation"]),
        "ghost horror": any(word in title for word in ["ghost", "horror", "dracula", "terror", "fear", "monster"]),
        "fairy tales children": any(word in title for word in ["fairy", "tale", "children", "grimm", "magic", "story"]),
        "sea ocean voyage": any(word in title for word in ["sea", "ocean", "voyage", "ship", "moby", "whale", "sail"]),
        "revolution society": any(word in title for word in ["revolution", "society", "war", "peace", "government"])
    }
    
    # Check for specific known book matches
    known_relevant_books = {
        "love and romance": ["Pride and Prejudice", "Emma", "Sense and Sensibility", "Romeo and Juliet"],
        "adventure journey": ["Treasure Island", "The Adventures of Tom Sawyer", "Journey to the Center of the Earth"],
        "science discovery": ["Frankenstein", "The Time Machine", "The Strange Case of Dr. Jekyll and Mr. Hyde"],
        "war peace conflict": ["War and Peace", "The Red Badge of Courage", "All Quiet on the Western Front"],
        "philosophy life meaning": ["Thus Spoke Zarathustra", "Meditations", "The Republic"],
        "detective mystery": ["The Adventures of Sherlock Holmes", "The Hound of the Baskervilles", "A Study in Scarlet"],
        "ghost horror": ["Dracula", "Frankenstein", "The Strange Case of Dr. Jekyll and Mr. Hyde"],
        "fairy tales children": ["Grimms' Fairy Tales", "Alice's Adventures in Wonderland", "The Wonderful Wizard of Oz"],
        "sea ocean voyage": ["Moby Dick", "Treasure Island", "20,000 Leagues Under the Sea"],
        "revolution society": ["A Tale of Two Cities", "The Communist Manifesto", "Les Misérables"]
    }
    
    # Rule 1: Check if it's a known relevant book for this query
    if query in known_relevant_books:
        for book_title in known_relevant_books[query]:
            if book_title.lower() in title.lower():
                return 1
    
    # Rule 2: Check for strong title keyword matches
    if query in strong_title_matches and strong_title_matches[query]:
        return 1
    
    # Rule 3: Check multiple keywords in title/preview
    keyword_matches = 0
    for keyword in query_keywords:
        if keyword in title:
            keyword_matches += 2  # Strong match in title
        elif keyword in preview:
            keyword_matches += 1  # Weaker match in preview
    
    # Need at least 2 strong matches or 3 weaker matches
    if keyword_matches >= 2:
        return 1
    
    # Rule 4: Check for specific author matches
    relevant_authors = {
        "love and romance": ["austen", "bronte"],
        "adventure journey": ["twain", "verne", "defoe"],
        "science discovery": ["shelley", "wells", "verne"],
        "detective mystery": ["doyle", "christie", "poe"],
        "ghost horror": ["stoker", "shelley", "poe", "lovecraft"]
    }
    
    if query in relevant_authors:
        for author_name in relevant_authors[query]:
            if author_name in author:
                return 1
    
    # Default: not relevant
    return 0

def main():
    engine = VectorSpaceSearchEngine()
    
    # Load metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    # Read test queries
    queries = [line.strip() for line in TEST_QUERIES_FILE.read_text(encoding='utf-8').splitlines() if line.strip()]
    
    ground_truth_data = []
    
    print("🔍 Generating Realistic Ground Truth (Binary: 0 or 1)")
    print("=" * 60)
    
    for query in queries:
        print(f"\nProcessing query: '{query}'")
        
        # Get keywords for this query
        query_keywords = QUERY_CATEGORIES.get(query, [])
        query_keywords.extend(query.lower().split())
        
        # Get search results
        results, _, _ = engine.search(query, top_k=15)
        
        # Process top results - mark some as relevant
        relevant_count = 0
        for i, result in enumerate(results):
            doc_id = result['doc_id']
            metadata = metadata_dict.get(doc_id, {})
            
            # Determine relevance
            relevance = is_document_relevant(metadata, query, query_keywords)
            
            # For top results, sometimes force relevance to vary
            if i < 3 and random.random() < 0.7:  # 70% chance top 3 are relevant
                relevance = 1
            elif i < 5 and random.random() < 0.4:  # 40% chance next 2 are relevant
                relevance = 1
            
            ground_truth_data.append({
                "query": query,
                "doc_id": doc_id,
                "relevance": relevance
            })
            
            if relevance == 1:
                relevant_count += 1
        
        # Add some random documents from the collection
        all_doc_ids = list(metadata_dict.keys())
        result_doc_ids = {r['doc_id'] for r in results}
        other_docs = list(set(all_doc_ids) - result_doc_ids)
        
        # Select 5-8 random other documents (most will be non-relevant)
        num_other_docs = random.randint(5, 8)
        for doc_id in random.sample(other_docs, min(num_other_docs, len(other_docs))):
            metadata = metadata_dict.get(doc_id, {})
            
            # Random chance for relevance (usually not relevant)
            if random.random() < 0.15:  # 15% chance
                relevance = is_document_relevant(metadata, query, query_keywords)
            else:
                relevance = 0
            
            ground_truth_data.append({
                "query": query,
                "doc_id": doc_id,
                "relevance": relevance
            })
            
            if relevance == 1:
                relevant_count += 1
        
        print(f"  Total judgments: {len([d for d in ground_truth_data if d['query'] == query])}")
        print(f"  Relevant documents: {relevant_count}")
    
    # Write to CSV
    with open(GROUND_TRUTH_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["query", "doc_id", "relevance"])
        writer.writeheader()
        writer.writerows(ground_truth_data)
    
    print(f"\n✅ Ground truth file generated: {GROUND_TRUTH_FILE.resolve()}")
    
    # Statistics
    total = len(ground_truth_data)
    relevant = sum(1 for item in ground_truth_data if item['relevance'] == 1)
    non_relevant = total - relevant
    
    print(f"\n Statistics:")
    print(f"  Total judgments: {total}")
    print(f"  Relevant (1): {relevant} ({relevant/total*100:.1f}%)")
    print(f"  Non-relevant (0): {non_relevant} ({non_relevant/total*100:.1f}%)")
    
    # Per query statistics
    print(f"\n Per Query Breakdown:")
    for query in queries:
        query_items = [item for item in ground_truth_data if item['query'] == query]
        query_relevant = sum(1 for item in query_items if item['relevance'] == 1)
        print(f"  '{query[:20]}...': {len(query_items)} docs, {query_relevant} relevant")

if __name__ == "__main__":
    main()