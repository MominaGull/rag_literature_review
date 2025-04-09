import os
from langchain_community.document_loaders import TextLoader


def get_project_root():
    """Get the absolute path to the project root directory."""
    # Current file is in /ingestion/pdf_parser.py, so go up two levels
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_all_markdowns(markdown_dir: str = None):
    """
    Parse all Markdown files in a directory.

    Args:
        markdown_dir (str, optional): The directory containing the Markdown files.
            If not provided, defaults to the 'data/raw_markdowns' directory in project root.

    Returns:
        list: A list of tuples containing (filename, pages) for each parsed Markdown.
    """
    parsed_markdowns = []

    # If markdown_dir is not provided or is relative, make it absolute
    if markdown_dir is None:
        markdown_dir = os.path.join(get_project_root(), "data")
    elif not os.path.isabs(markdown_dir):
        markdown_dir = os.path.join(get_project_root(), markdown_dir)

    # Check if directory exists
    if not os.path.exists(markdown_dir):
        raise FileNotFoundError(
            f"Markdown directory not found: {markdown_dir}")

    for filename in os.listdir(markdown_dir):
        if filename.lower().endswith(".mmd"):
            filepath = os.path.join(markdown_dir, filename)
            loader = TextLoader(filepath)
            pages = loader.lazy_load()

            parsed_markdowns.append((filename, pages))
    print(parsed_markdowns)

    return parsed_markdowns
