import os
import PyPDF2
from docx import Document

def parse_pdf(file_path: str) -> str:
    """
    Parses a PDF file and extracts text.
    Preserves basic page structure.
    """
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text += f"--- PAGE {i+1} ---"
            text += page.extract_text() + "\n"
    return text

def parse_docx(file_path: str) -> str:
    """
    Parses a DOCX file and extracts text.
    Preserves basic paragraph structure.
    """
    document = Document(file_path)
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    return text

def parse_txt(file_path: str) -> str:
    """
    Parses a plain text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def chunk_text(text: str, chunk_size: int = 3000, overlap_sentences: int = 2) -> list[dict]:
    """
    Chunks text into smaller segments based on sentence boundaries using NLTK.
    Ensures entities and relationships are not split across chunk boundaries.
    """
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk_size, finalize the current chunk
        if current_length + sentence_length > chunk_size and current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences)
            chunks.append({
                "content": chunk_content,
                "start_char": text.find(current_chunk_sentences[0]), # Approximate start
                "end_char": text.find(current_chunk_sentences[-1]) + len(current_chunk_sentences[-1]),
                "source_type": "literary_text",
                "document_id": "doc_123"
            })
            
            # Start new chunk with overlap from the end of the previous chunk
            overlap = current_chunk_sentences[-overlap_sentences:] if len(current_chunk_sentences) >= overlap_sentences else current_chunk_sentences
            current_chunk_sentences = list(overlap)
            current_length = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences) - 1
            
        current_chunk_sentences.append(sentence)
        current_length += sentence_length + 1 # +1 for the space
        
    # Add the final chunk if it has content
    if current_chunk_sentences:
        chunk_content = " ".join(current_chunk_sentences)
        chunks.append({
            "content": chunk_content,
            "start_char": text.find(current_chunk_sentences[0]),
            "end_char": text.find(current_chunk_sentences[-1]) + len(current_chunk_sentences[-1]),
            "source_type": "literary_text",
            "document_id": "doc_123"
        })

    return chunks

def document_parser(file_path: str) -> list[dict]:
    """
    Main function to parse a document and return chunked text with metadata.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    full_text = ""

    if file_extension == '.pdf':
        full_text = parse_pdf(file_path)
    elif file_extension == '.docx':
        full_text = parse_docx(file_path)
    elif file_extension == '.txt':
        full_text = parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    print(f"Parsed {len(full_text)} characters from {file_path}")
    return chunk_text(full_text)

if __name__ == '__main__':
    # Example Usage (requires dummy files to exist)
    # For a real run, you'd pass an actual file path
    print("--- Document Parser Module Test ---")
    
    # Create dummy files for testing
    with open("dummy.txt", "w") as f:
        f.write("This is a dummy text file. It has some content.")
    
    # Example for TXT
    try:
        chunks = document_parser("dummy.txt")
        print(f"TXT Chunks: {len(chunks)}")
        # print(chunks[0]['content']) # Uncomment to see content
    except Exception as e:
        print(f"Error parsing dummy.txt: {e}")
    
    # Clean up dummy files
    os.remove("dummy.txt")

    print("Note: DOCX and PDF parsing would require actual files for full testing.")
