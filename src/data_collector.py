# -*- coding: utf-8 -*-
"""
Data Collector Module for Jost-o-Joo Search Engine

This module handles document metadata extraction from Project Gutenberg text files.
It scans the documents directory, extracts titles, authors, and other metadata,
and saves it to a JSON file for use by the search engine

Created on Sat Dec 13 19:58:22 2025

@author: Rihem Touzi
"""


import os
import json
import re
from pathlib import Path
from datetime import datetime

# Base directory is set to project root (Jost-o-Joo/)
BASE_DIR = Path(__file__).resolve().parent.parent

class DataCollector:
    def __init__(self,
             documents_dir=BASE_DIR / "data" / "documents",
             metadata_file=BASE_DIR / "data" / "metadata.json"):
        self.documents_dir = Path(documents_dir)
        self.metadata_file = Path(metadata_file)
        #metadata (dict): In-memory storage of document metadata
        self.metadata = {}
        
        # Manual mapping of known book IDs to titles
        # This helps when title extraction from content fails
        self.book_titles = {
            1342: "Pride and Prejudice",
            84: "Frankenstein",
            11: "Alice's Adventures in Wonderland",
            1661: "The Adventures of Sherlock Holmes",
            98: "A Tale of Two Cities",
            74: "The Adventures of Tom Sawyer",
            2701: "Moby Dick",
            76: "Adventures of Huckleberry Finn",
            2591: "Grimms' Fairy Tales",
            64317: "The Great Gatsby",
            1080: "A Modest Proposal",
            2554: "Crime and Punishment",
            174: "The Picture of Dorian Gray",
            345: "Dracula",
            3207: "Leviathan",
            43: "The Strange Case of Dr. Jekyll and Mr. Hyde",
            1232: "Leaves of Grass",
            1400: "Great Expectations",
            1952: "The Yellow Wallpaper",
            1260: "The Journals of Lewis and Clark",
            408: "The Souls of Black Folk",
            120: "Treasure Island",
            160: "The Awakening",
            1184: "The Count of Monte Cristo",
            2542: "A Doll's House",
            219: "Heart of Darkness",
            28054: "The Brothers Karamazov",
            25344: "The Scarlet Letter",
            5200: "Metamorphosis",
            46: "A Christmas Carol",
            158: "Emma",
            161: "Sense and Sensibility",
            1998: "Thus Spoke Zarathustra",
            30254: "Don Quixote",
            2600: "War and Peace",
            205: "The Age of Innocence",
            244: "A Study in Scarlet",
            4085: "The Republic",
            1497: "The Republic",
            768: "Wuthering Heights",
            164: "The Importance of Being Earnest",
            23: "The Iliad",
            18247: "The Last of the Mohicans",
            1399: "The Hound of the Baskervilles",
            2814: "Dubliners",
            4300: "Meditations",
            19942: "Candide",
            215: "The Call of the Wild",
            2590: "Guy Mannering",
            20228: "The Prince"
        }
        
        # Ensure data directory exists
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    def extract_book_id_from_content(self, content):
        """
        Extract Gutenberg book ID from document header.
        
        Searches for patterns like "PROJECT GUTENBERG EBOOK 12345" in the
        first 1000 characters of the document.
        
        Args:
            content (str): Document content
            
        Returns:
            int or None: Book ID if found, None otherwise
        """
        patterns = [
            r'PROJECT GUTENBERG EBOOK (\d+)',
            r'GUTENBERG EBOOK (\d+)',
            r'EBOOK (\d+)',
            r'FILE (\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content[:1000], re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        return None
    
    def extract_title_from_content(self, content):
        """Extract title from document content using multiple strategies"""
        # Strategy 1: Look for Title: pattern
        title_patterns = [
            r'Title:\s*(.+?)\n',
            r'Title:\s*(.+?)\r',
            r'TITLE:\s*(.+?)\n',
            r'Title\s*:\s*(.+?)\n',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content[:5000], re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 5 and "project gutenberg" not in title.lower():
                    # Clean up the title
                    title = re.sub(r'[\*\#\_]', '', title)  # Remove special chars
                    title = re.sub(r'\s+', ' ', title)  # Normalize spaces
                    return title[:200]  # Limit length
        
        # Strategy 2: Look for book ID in Gutenberg header
        book_id = self.extract_book_id_from_content(content)
        if book_id and book_id in self.book_titles:
            return self.book_titles[book_id]
        
        # Strategy 3: Look for first meaningful line after Gutenberg header
        lines = content.split('\n')
        in_header = True
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip Gutenberg header lines
            if 'project gutenberg' in line.lower() or 'ebook' in line.lower():
                continue
            
            # Skip lines that are too short or look like metadata
            if len(line) < 20:
                continue
            
            # Skip lines that start with special characters
            if line.startswith(('*', '#', '_', '[', '(')):
                continue
            
            # This might be the actual title or first line of content
            # Clean it up
            line = re.sub(r'[^\w\s\-\'\",\.!?]', '', line)
            if len(line) > 10:
                return line[:150]
        
        # Strategy 4: Extract from filename patterns in content
        ebook_patterns = [
            r'GUTENBERG EBOOK OF (.+?) BY',
            r'GUTENBERG EBOOK (.+?)\.',
            r'EBOOK OF (.+?)\.',
        ]
        
        for pattern in ebook_patterns:
            match = re.search(pattern, content[:2000], re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 5:
                    title = re.sub(r'[\*\#\_]', '', title)
                    return title[:200]
        
        return "Unknown Document"
    
    def get_document_title(self, doc_id, content):
        """Get title for a document with fallback strategies"""
        # Try to extract from content
        title = self.extract_title_from_content(content)
        
        # If still generic, create a descriptive title
        if title == "Unknown Document" or len(title) < 5:
            # Look for author name
            author_match = re.search(r'Author:\s*(.+?)\n', content[:2000], re.IGNORECASE)
            author = f" by {author_match.group(1).strip()}" if author_match else ""
            
            # Count words to guess type
            word_count = len(content.split())
            if word_count > 50000:
                doc_type = "Novel"
            elif word_count > 10000:
                doc_type = "Story"
            else:
                doc_type = "Document"
            
            title = f"{doc_type}{author}"
        
        return title
    
    def collect_metadata(self):
        """Collect metadata from all documents in the directory"""
        print(f"📁 Scanning directory: {self.documents_dir.absolute()}")
        
        documents = sorted([f for f in self.documents_dir.glob("Doc_*.txt")])
        
        if not documents:
            print("❌ No documents found! Please run downloadTxtFiles.py first.")
            return False
        
        print(f" Found {len(documents)} documents")
        
        for doc_path in documents:
            try:
                doc_id = doc_path.stem  # "Doc_1", "Doc_2", etc.
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(100000)  # Read first 100k chars for speed
                
                # Get title
                title = self.get_document_title(doc_id, content)
                
                # Extract author if available
                author = "Unknown"
                author_match = re.search(r'Author:\s*(.+?)\n', content, re.IGNORECASE)
                if author_match:
                    author = author_match.group(1).strip()
                
                # Extract some sample content for preview (skip headers)
                lines = content.split('\n')
                preview_lines = []
                for line in lines:
                    line = line.strip()
                    if (line and 
                        'project gutenberg' not in line.lower() and
                        'ebook' not in line.lower() and
                        'title:' not in line.lower() and
                        'author:' not in line.lower() and
                        not line.startswith('***')):
                        preview_lines.append(line)
                        if len('\n'.join(preview_lines)) > 200:
                            break
                
                preview = ' '.join(preview_lines[:3])[:200]
                if len(content) > 100000:
                    preview += "..."
                
                # Get word count (only words, skip headers)
                clean_content = self._remove_gutenberg_header(content)
                words = self._clean_text(clean_content)
                word_count = len(words.split())
                
                # Store metadata
                self.metadata[doc_id] = {
                    "id": doc_id,
                    "title": title,
                    "author": author,
                    "filename": doc_path.name,
                    "path": str(doc_path.absolute()),
                    "word_count": word_count,
                    "char_count": len(content),
                    "created": datetime.now().isoformat(),
                    "preview": preview,
                    "source": "Project Gutenberg",
                    "language": "English"
                }
                
                print(f"  ✓ {doc_path.name}: '{title[:60]}...' ({word_count} words)")
                
            except Exception as e:
                print(f"  ✗ Error processing {doc_path.name}: {e}")
        
        # Save metadata to JSON file
        self._save_metadata()
        return True
    
    def _remove_gutenberg_header(self, content):
        """Remove Project Gutenberg header from content"""
        # Find where the actual book starts
        end_markers = [
            "*** START OF",
            "***START OF",
            "Produced by",
            "Beginning of",
            "START OF THIS"
        ]
        
        for marker in end_markers:
            idx = content.find(marker)
            if idx != -1:
                # Find the next line break after marker
                content = content[idx:]
                break
        
        return content
    
    def _clean_text(self, text):
        """Basic text cleaning for word count"""
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"\n Metadata saved to: {self.metadata_file.absolute()}")
        print(f" Total documents: {len(self.metadata)}")
        
        # Print summary
        print("\n Title Summary:")
        titles_with_words = [(meta['title'], meta['word_count']) for meta in self.metadata.values()]
        titles_with_words.sort(key=lambda x: x[1], reverse=True)
        
        for i, (title, word_count) in enumerate(titles_with_words[:10]):
            print(f"  {i+1:2d}. {title[:50]:50} ({word_count:6,} words)")
    
    def get_document_content(self, doc_id):
        """Get content of a specific document"""
        if doc_id in self.metadata:
            doc_path = Path(self.metadata[doc_id]['path'])
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        return None

def main():
    """Main function to run data collection"""
    print("=" * 60)
    print(" DATA COLLECTOR MODULE")
    print("=" * 60)
    
    collector = DataCollector()
    
    if collector.collect_metadata():
        print("\n✅ Data collection completed successfully!")
        
        # Show improved sample
        print("\n Sample Improved Titles:")
        for i in range(1, 6):
            doc_id = f"Doc_{i}"
            if doc_id in collector.metadata:
                meta = collector.metadata[doc_id]
                print(f"\n  Document: {doc_id}")
                print(f"    Title: {meta['title']}")
                print(f"    Author: {meta.get('author', 'Unknown')}")
                print(f"    Words: {meta['word_count']}")
    else:
        print("❌ Data collection failed!")

if __name__ == "__main__":
    main()