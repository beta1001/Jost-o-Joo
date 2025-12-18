# -*- coding: utf-8 -*-

# fix_titles.py
import json
from pathlib import Path

def fix_metadata_titles():
    metadata_file = Path("../data/metadata.json")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Manual title assignments based on what we know
    title_map = {
        "Doc_1": "Pride and Prejudice",
        "Doc_2": "Frankenstein",
        "Doc_3": "Alice's Adventures in Wonderland",
        "Doc_4": "The Adventures of Sherlock Holmes",
        "Doc_5": "A Tale of Two Cities",
        "Doc_6": "The Adventures of Tom Sawyer",
        "Doc_7": "Moby Dick",
        "Doc_8": "Adventures of Huckleberry Finn",
        "Doc_9": "Grimms' Fairy Tales",
        "Doc_10": "The Great Gatsby",
        "Doc_11": "A Modest Proposal",
        "Doc_12": "Crime and Punishment",
        "Doc_13": "The Picture of Dorian Gray",
        "Doc_14": "Dracula",
        "Doc_15": "Leviathan",
        "Doc_16": "The Strange Case of Dr. Jekyll and Mr. Hyde",
        "Doc_17": "Leaves of Grass",
        "Doc_18": "Great Expectations",
        "Doc_19": "The Yellow Wallpaper",
        "Doc_20": "The Journals of Lewis and Clark",
        "Doc_21": "The Souls of Black Folk",
        "Doc_22": "Treasure Island",
        "Doc_23": "The Awakening",
        "Doc_24": "The Count of Monte Cristo",
        "Doc_25": "A Doll's House",
        "Doc_26": "Heart of Darkness",
        "Doc_27": "The Brothers Karamazov",
        "Doc_28": "The Scarlet Letter",
        "Doc_29": "Metamorphosis",
        "Doc_30": "A Christmas Carol",
        "Doc_31": "Emma",
        "Doc_32": "Sense and Sensibility",
        "Doc_33": "Thus Spoke Zarathustra",
        "Doc_34": "Don Quixote",
        "Doc_35": "War and Peace",
        "Doc_36": "The Age of Innocence",
        "Doc_37": "A Study in Scarlet",
        "Doc_38": "The Republic",
        "Doc_39": "The Republic (Plato)",
        "Doc_40": "Wuthering Heights",
        "Doc_41": "The Importance of Being Earnest",
        "Doc_42": "The Iliad",
        "Doc_43": "The Last of the Mohicans",
        "Doc_44": "The Hound of the Baskervilles",
        "Doc_45": "Dubliners",
        "Doc_46": "Meditations",
        "Doc_47": "Candide",
        "Doc_48": "The Call of the Wild",
        "Doc_49": "Guy Mannering",
        "Doc_50": "The Prince"
    }
    
    # Update metadata with proper titles
    for doc_id, title in title_map.items():
        if doc_id in metadata:
            metadata[doc_id]["title"] = title
            print(f"✓ Updated {doc_id}: {title}")
    
    # Save updated metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Updated titles in {metadata_file}")
    print(f"📊 Total documents: {len(metadata)}")

if __name__ == "__main__":
    fix_metadata_titles()