import os
from langchain_community.document_loaders import PyPDFLoader


def get_project_root():
    """Get the absolute path to the project root directory."""
    # Current file is in /ingestion/pdf_parser.py, so go up two levels
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_all_pdfs(pdf_dir: str = None):
    """
    Parse all PDF files in a directory.

    Args:
        pdf_dir (str, optional): The directory containing the PDF files.
            If not provided, defaults to the 'data/raw_pdfs' directory in project root.

    Returns:
        list: A list of tuples containing (filename, pages) for each parsed PDF.
    """
    parsed_pdfs = []
    
    # If pdf_dir is not provided or is relative, make it absolute
    if pdf_dir is None:
        pdf_dir = os.path.join(get_project_root(), "data")
    elif not os.path.isabs(pdf_dir):
        pdf_dir = os.path.join(get_project_root(), pdf_dir)
        
    # Check if directory exists
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.lazy_load()

            parsed_pdfs.append((filename, pages))
    
    return parsed_pdfs