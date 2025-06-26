import torch
import os
from pathlib import Path


def parse_folder(directory: str):
    """Returns all files in the directory recursively"""
    path = Path(directory)
    files = []
    for file_path in path.rglob('*'):
        if file_path.is_file():
            files.append(file_path)  # Keep as Path object
    return files
    

def get_suffix(files, suffix):
    """Filter files by suffix"""
    suffix_files = []
    for file in files:
        if file.suffix.lower() == f".{suffix}":
            suffix_files.append(file)
    return suffix_files


def parse_txt(file_paths):
    """Parse text files and return their content"""
    texts = []
    for file in file_paths:
        try:
            with open(file, "r", encoding="utf-8") as f:
                texts.append(clean_text(f.read()))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return texts
            

def parse_pdfs(file_paths):
    """Parse PDF files and return their text content"""
    import PyPDF3
    text_chunks = []
    for file in file_paths:
        try:
            with open(file, "rb") as f:  # Fixed: use 'file' not 'file_paths'
                reader = PyPDF3.PdfFileReader(f)
                file_text = ""
                for page in reader.pages:
                    file_text += page.extractText() or " "
                text_chunks.append(clean_text(file_text))
        except Exception as e:
            print(f"Error reading PDF {file}: {e}")
    return text_chunks



def clean_text(text: str) -> str:
    import re
    
    if not text:
        return ""
    
    text = text.replace("\u00a0", " ")
    
    lines = text.splitlines()
    good_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if len(line) <= 2:
            continue
            
        if len(line) < 10 and line.isupper():
            continue
            
        line = re.sub(r'[ \t]+', ' ', line)
        good_lines.append(line)
    
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'\b([A-Z])\s+([A-Z])\s+([A-Z])', r'\1\2\3', text)
    
    return text.strip()


def parse_directory(directory: str):
    all_files = parse_folder(directory)
    
    txt_files = get_suffix(all_files, "txt")
    pdf_files = get_suffix(all_files, "pdf")
    
    all_texts = []
    
    if txt_files:
        print(f"Found {len(txt_files)} text files")
        txt_content = parse_txt(txt_files)
        all_texts.extend(txt_content)
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files")
        pdf_content = parse_pdfs(pdf_files)
        all_texts.extend(pdf_content)
    
    return all_texts


def count_words(texts):
    """
    Count all words from a list of text strings
    Returns a dictionary with word counts and total statistics
    """
    from collections import Counter
    import re
    
    word_counts = Counter()
    total_words = 0
    total_chars = 0
    
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        word_counts.update(words)
        total_words += len(words)
        total_chars += len(text)
    
    return {
        'word_counts': word_counts,
        'total_words': total_words,
        'unique_words': len(word_counts),
        'total_characters': total_chars,
        'most_common': word_counts.most_common(10)
    }


def print_word_statistics(word_stats):
    """Print formatted word statistics"""
    print(f"\n--- WORD STATISTICS ---")
    print(f"Total words: {word_stats['total_words']:,}")
    print(f"Unique words: {word_stats['unique_words']:,}")
    print(f"Total characters: {word_stats['total_characters']:,}")
    print(f"\nTop 10 most common words:")
    for i, (word, count) in enumerate(word_stats['most_common'], 1):
        print(f"{i:2d}. {word:<15} {count:,} times")


if __name__ == "__main__":
    directory = "director_random" 
    texts = parse_directory(directory)
    print(f"Extracted text from {len(texts)} files")
    
    word_stats = count_words(texts)
    print_word_statistics(word_stats)

    print(len(texts))
    if texts:
        print(f"\nSample cleaned text (first 200 chars):")
        print(repr(texts[1][:200]))
    