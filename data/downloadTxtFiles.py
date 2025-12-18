import requests
import os
from concurrent.futures import ThreadPoolExecutor

def download_gutenberg_documents(output_dir="data/documents", max_documents=50):
    """
    Download 50 documents from Project Gutenberg as .txt files
    
    Parameters:
    - output_dir: Directory where files will be saved (relative path)
    - max_documents: Number of documents to download (default: 50)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Project Gutenberg books - guaranteed .txt format
    gutenberg_books = [
        # Classic novels
        1342, 84, 11, 1661, 98, 74, 2701, 76, 2591, 64317,
        # More classics
        1080, 2554, 174, 345, 3207, 43, 1232, 1400, 1952, 1260,
        # Shakespeare and plays
        408, 120, 160, 1184, 2542, 219, 28054, 25344, 5200, 46,
        # Philosophy and essays
        158, 161, 1998, 30254, 2600, 205, 244, 4085, 1497, 768,
        # Science and adventure
        164, 23, 18247, 1399, 2814, 4300, 19942, 215, 2590, 20228,
        # Additional books to ensure 50
        145, 135, 1400, 160, 1727, 76, 1080, 4300, 28054, 25344
    ]
    
    def download_book(book_id, index):
        """Download a single book as .txt file"""
        urls = [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            f"http://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Clean the text (remove Gutenberg headers/footers)
                    text = response.text
                    # Find start of actual book (after *** START OF ... ***)
                    start_markers = ["*** START OF", "***START OF", "Produced by"]
                    for marker in start_markers:
                        idx = text.find(marker)
                        if idx != -1:
                            # Find next line break after marker
                            text = text[idx:]
                            break
                    
                    # Save as .txt file with your naming convention
                    filename = os.path.join(output_dir, f"Doc_{index}.txt")
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(text[:50000])  # First 50k chars for manageable size
                    
                    print(f"✓ Downloaded Doc_{index}.txt (Book ID: {book_id})")
                    return True
            except Exception as e:
                continue
        
        print(f"✗ Failed to download book {book_id}")
        return False
    
    # Download all books in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, book_id in enumerate(gutenberg_books[:max_documents], 1):
            futures.append(executor.submit(download_book, book_id, i))
        
        for future in futures:
            if future.result():
                successful += 1
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESSFULLY DOWNLOADED {successful}/{max_documents} DOCUMENTS")
    print(f"📁 Location: {os.path.abspath(output_dir)}")
    print(f"📄 Files: Doc_1.txt through Doc_{successful}.txt")
    print(f"{'='*60}")
    
    # List all downloaded files
    print("\n📋 Downloaded files:")
    for i in range(1, successful + 1):
        file_path = os.path.join(output_dir, f"Doc_{i}.txt")
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  - Doc_{i}.txt ({file_size:,} bytes)")
    
    return successful

# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    # Download to data/documents folder
    download_gutenberg_documents(output_dir="./documents")