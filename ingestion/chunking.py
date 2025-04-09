from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document


def chunk_documents(document, chunk_size=5000,
                    chunk_overlap=500):
    """
    Splits a list of Document objects into smaller chunks.

    Args:
        documents (list[Document]): List of Document objects to split
        chunk_size (int, optional): Maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 100.

    Returns:
        list[Document]: A new list of Document objects with chunked content
    """
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     separators=["\n\n", "\n", " ", ""]
    # )

    # chunked_doc = []
    # for chunk in splitter.split_text(document.page_content):
    #     # Create a new Document for each chunk
    #     chunked_doc.append(Document(
    #         page_content=chunk,
    #         metadata=document.metadata
    #     ))
    # print("chunked_doc", chunked_doc)
    # return chunked_doc

    headers_to_split_on = [
        ("#", "Section"),
        ("##", "Subsection"),
        ("###", "Subsubsection"),
        ("####", "Subsubsubsection")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    structured_docs = markdown_splitter.split_text(document.page_content)
    with open("structured_docs.txt", "w") as f:
        for sd in structured_docs:
            f.write("New Section\n")
            f.write(f"{sd.metadata}\n")
            f.write(f"{sd.page_content}\n")

    # Step 2: For each structured doc, further chunk if needed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    final_chunks = []
    for structured_doc in structured_docs:
        # structured_doc is also a Document
        # containing partial text + metadata like { 'Section': '## Methods' }
        sub_chunks = text_splitter.split_text(structured_doc.page_content)
        for sc in sub_chunks:
            final_chunks.append(Document(
                page_content=sc,
                metadata={**document.metadata, **structured_doc.metadata, "text": sc}
            ))
    with open("final_docs.txt", "w") as f:
        for fc in final_chunks:
            f.write("New chunk\n")
            f.write(f"{fc.metadata}\n")
            f.write(f"{fc.page_content}\n")

    print("final_chunks", final_chunks)
    return final_chunks
