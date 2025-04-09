from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def chunk_documents(document, chunk_size=1000,
                    chunk_overlap=100):
    """
    Splits a list of Document objects into smaller chunks.

    Args:
        documents (list[Document]): List of Document objects to split
        chunk_size (int, optional): Maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 100.

    Returns:
        list[Document]: A new list of Document objects with chunked content
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_doc = []
    for chunk in splitter.split_text(document.page_content):
        # Create a new Document for each chunk
        chunked_doc.append(Document(
            page_content=chunk,
            metadata=document.metadata
        ))
    print("chunked_doc", chunked_doc)
    return chunked_doc
