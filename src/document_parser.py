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

def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> list[dict]:
    """
    Chunks text into smaller segments and adds basic metadata.
    For a real system, this would involve more sophisticated chunking
    and metadata extraction (e.g., chapter, section detection).
    """
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        chunk_content = text[current_pos:end_pos]
        chunks.append({
            "content": chunk_content,
            "start_char": current_pos,
            "end_char": end_pos,
            "source_type": "literary_text", # Example metadata
            "document_id": "doc_123" # Placeholder
        })
        current_pos += chunk_size - overlap
        if chunk_size - overlap <= 0 and current_pos < len(text):
            # Avoid infinite loop if overlap is too large or chunk_size too small
            current_pos += chunk_size

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
