import os
import base64
import json
import logging
import datetime
import tempfile
from io import BytesIO
import anthropic
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from dotenv import load_dotenv

load_dotenv(override=True)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_project_root():
    """Get the absolute path to the project root directory."""
    # Current file is in /ingestion/pdf_parser.py, so go up two levels
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
log_dir = os.path.join(get_project_root(), "logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"pdf_parser_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("pdf_parser")

# Create a separate log file for image descriptions
image_descriptions_log = os.path.join(
    log_dir, f"image_descriptions_{timestamp}.json")
image_data = {}

prompt = '''You are a scientific research assistant. You are provided with an image extracted from a scientific research paper titled: "{paper_name}".

Your task is to analyze the image carefully and provide a clear, concise, and accurate textual description that would help a researcher understand the image without seeing it.

Focus on:
- The type of visual (e.g., chart, graph, schematic, micrograph, etc.)
- The main components or features shown
- The relationships or trends visible (if any)
- The purpose of the image in the context of the research topic

Avoid:
- Overly generic descriptions
- If the image is not clear or does not provide significant information, state that explicitly.
- Do not add any extra headings with the response.

Keep the language scientific but accessible.
'''


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


def parse_create_markdown(pdf_dir: str = None):

    logger.info("Starting PDF to markdown conversion and image description")

    # If pdf_dir is not provided or is relative, make it absolute
    if pdf_dir is None:
        pdf_dir = os.path.join(get_project_root(), "data")
    elif not os.path.isabs(pdf_dir):
        pdf_dir = os.path.join(get_project_root(), pdf_dir)

    # Check if directory exists
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    logger.info(f"Looking for PDFs in directory: {pdf_dir}")

    processed_files = []
    markdown_contents = {}

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_markdown_dir:
        logger.info(f"Created temporary directory: {temp_markdown_dir}")

        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith(".pdf"):
                logger.info(f"Processing PDF: {filename}")

                # Create dictionary to store image data for this PDF
                pdf_data = {
                    "pdf_name": filename,
                    "images": {}
                }

                try:

                    converter = PdfConverter(
                        artifact_dict=create_model_dict(),)

                    filepath = os.path.join(pdf_dir, filename)
                    logger.info(f'Converting PDF to markdown: {filepath}')
                    rendered = converter(filepath)
                    text, _, images = text_from_rendered(rendered)

                    temp_markdown_name = filename.replace(".pdf", ".md")
                    temp_markdown_path = os.path.join(temp_markdown_dir, temp_markdown_name)
                    logger.info(f'Creating temporary markdown: {temp_markdown_path}')

                    # Write initial markdown content to temp file
                    with open(temp_markdown_path, "w", encoding="utf-8") as md_file:
                        md_file.write(text)

                    # Read the markdown content
                    with open(temp_markdown_path, "r", encoding="utf-8") as f:
                        markdown_content = f.readlines()

                    # Create a new list to store updated content
                    logger.info(
                        f'Updating markdown with image descriptions: {temp_markdown_name}')
                    updated_content = []
                    processed_keys = set()

                    # Process each line of the markdown file
                    for line in markdown_content:
                        # Check if any image key is in this line
                        matched_key = None
                        for key in images.keys():
                            if key in line and key not in processed_keys:
                                matched_key = key
                                processed_keys.add(key)
                                break

                        # If we found a match, add the description before the line
                        if matched_key:
                            logger.info(
                                f"Processing image in markdown: {matched_key}")

                            # Convert PIL image to base64
                            img = images[matched_key]
                            buffered = BytesIO()
                            img.save(buffered, format="JPEG")
                            base64_image = base64.b64encode(
                                buffered.getvalue()).decode('utf-8')

                            client = OpenAI()
                            try:
                                message = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "user",
                                         "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                                }
                                            },
                                            {"type": "text", "text": prompt.format(paper_name=filename)}
                                        ]}
                                    ]
                                )

                                description = message.choices[0].message.content
                                print(message)
                                logger.info(
                                    f"Generated description for {matched_key}")

                                # Log image data
                                pdf_data["images"][matched_key] = {
                                    "image_key": matched_key,
                                    "image_size": f"{img.width}x{img.height}",
                                    "description": description
                                }

                                # Add the description and then the line with the image
                                updated_content.append(f"\n{description}\n\n")
                                updated_content.append(line)
                            except Exception as e:
                                logger.error(
                                    f"Error generating description for {matched_key}: {str(e)}")
                                # Add just the line without description
                                updated_content.append(line)
                        else:
                            # Just add the line as-is
                            updated_content.append(line)

                    # Store processed content in memory
                    markdown_contents[filename] = ''.join(updated_content)
                    processed_files.append(filename)

                    # For debugging - write the updated content to the temp file
                    with open(temp_markdown_path, "w", encoding="utf-8") as f:
                        f.writelines(updated_content)
                    
                    logger.info(f"Processed {len(processed_keys)} images in {filename}")

                    # Check if any images weren't found
                    missing_keys = set(images.keys()) - processed_keys
                    if missing_keys:
                        logger.warning(
                            f"{len(missing_keys)} images not found in the markdown for {filename}")
                        for key in missing_keys:
                            logger.warning(f"  - Missing image: {key}")

                    # Add the PDF data to our image data collection
                    image_data[filename] = pdf_data

                    # Save the image data after each PDF in case of failures
                    with open(image_descriptions_log, 'w', encoding='utf-8') as f:
                        json.dump(image_data, f, indent=2)

                except Exception as e:
                    logger.error(f"Error processing PDF {filename}: {str(e)}")

        logger.info(f"Completed processing all PDFs. Log saved to: {log_file}")
        logger.info(f"Image descriptions saved to: {image_descriptions_log}")

    logger.info(f"Completed processing {len(processed_files)} PDFs.")
    logger.info(f"Log saved to: {log_file}")
    logger.info(f"Image descriptions saved to: {image_descriptions_log}")

    return processed_files, markdown_contents


# if __name__ == "__main__":
#     parse_create_markdown()
