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
    
    
# Error reading PDF director_random/others/Instrument Engineers Handbook, Volume 3 Process Software and Digital Networks, Fourth Edition (Volume 3).pdf: Stream has ended unexpectedly
# PdfReadWarning: Xref table not zero-indexed. ID numbers for objects will be corrected. [pdf.py:1865]
# Error reading PDF director_random/books/Pattern Recognition, Bishop.pdf: Unexpected escaped string: b'y'
# PdfReadWarning: Superfluous whitespace found in object header b'\xf9z\xceh\xcbK\x1bW\x95>\xd1\x1d\xb2@1GG\xe5](\xb0\xe4\x87\xfeF7\xf5\xc6\x82\xf1\x14\xc6\x1f,iQ\xd7\x8a\xd2\xbf\xf1a\xfe\xb8\xd2-\x1c\x9e[\xd2\xe8\x19W\xe2\xed\xfb\xc3\xfdrT\x16\xd9\x0e\x87\xa0X\x97\x92\xa8~\xc8\xfd\xa3\xe3\x8c\x05\xa8{o\x97\xbf.\xb4MGN[\xdb\x88\x1d\xa5%\xf7\x128\xe8H\x1d\x0elp\xe1\x89\x8d\xb7DXb\xf2\xf9_M\xb6_U\xd0\xa0\xe9Vr\x07\xe2r\xbe\x00\xb4\x81}\'K\x07\xe1+O\xf8\xc9\xfd\xb8\xd0H\x01\xbf\xd1\x1aa\xf0\xff\x00\x91\x87\xfa\xe3Ah6\xban\x9f\x0e\xd1\x91\xbf_\x8f\xfbq\x00)^"\xb5_\x842\xd3\xfdl\x92\xa3m,m.\x89QF\xa0\xae\xcd\xfd0\x04\x90\x90jj\xd6\x97f\xde\xd8\x11\x18\xe3M\xab\xd4x\xe4J\x194\xd6K*\xf1*O\xdf\x93!4\xa0\x9ae\xaa\x8aL8\xb7\x81jm\x82\x90B\x07\xd1\xb4\xe4G%\xa5\x7f\x9b\x1bP\xbf\xd0\xb4\xfee\xff\x00\x82\xc1i\xa0\xb4\xdb\xd9\xff\x002\xff\x00\xc1\xff\x00n6\x8a' b'' [pdf.py:1790]
# Error reading PDF director_random/books/Artificial Neural Networks and Machine Learning – ICANN 2018_ 27th International Conference on Artificial Neural Networks, Rhodes, Greece, October 4-7, 2018, Proceedings, Part I ( PDFDrive ).pdf: invalid literal for int() with base 10: b'\xf9z\xceh\xcbK\x1bW\x95>\xd1\x1d\xb2@1GG\xe5](\xb0\xe4\x87\xfeF7\xf5\xc6\x82\xf1\x14\xc6\x1f,iQ\xd7\x8a\xd2\xbf\xf1a\xfe\xb8\xd2-\x1c\x9e[\xd2\xe8\x19W\xe2\xed\xfb\xc3\xfdrT\x16\xd9\x0e\x87\xa0X\x9
# Error reading PDF director_random/books/Valliappa Lakshmanan, Marco Tranquillin, Firat Tekiner - Architecting Data and Machine Learning Platforms_ Enable Analytics and AI-Driven Innovation in the Cloud-O'Reilly Media (2023).pdf: '/Contents'
# Error reading PDF director_random/books/V Kishore Ayyadevara, Yeshwanth Reddy - Modern Computer Vision with PyTorch_ A Practical Roadmap From Deep Learning Fundamentals to Advanced Applications and Generative AI, 2nd Edition-Packt Publishin.pdf: EOF marker not found
# Error reading PDF director_random/books/Architecting_Data_and_Machine_Learning_Platforms.pdf: '/Contents'
# Error reading PDF director_random/books/Eric Matthes - Python Crash Course-No Starch Press (2023).pdf: '/Contents'
# Error reading PDF director_random/books/(Adaptive Computation and Machine Learning series) Bengio, Yoshua_ Courville, Aaron_ Goodfellow, Ian J - Deep learning_ adaptive computation and machine learning-The MIT Press (2016).pdf: Stream has ended unexpectedly
# Error reading PDF director_random/books/Michael Munn, David Pitman - Explainable AI for Practitioners_ Designing and Implementing Explainable ML Solutions-O'Reilly Media (2022).pdf: '/Contents'
# Extracted text from 105 files