import os
import sys
import tempfile
import streamlit as st
import shutil

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ingest_pipeline import main as run_ingestion
from retrieval_chain import build_retrieval_chain

from dotenv import load_dotenv
load_dotenv(override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")

def main():
    st.title("RAG Chatbot for Research Papers")

    # Sidebar with PDF upload and ingestion options
    st.sidebar.title("Document Management")

    # Add file uploader to sidebar
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    # User query input
    st.subheader("Ask a question about the ingested PDFs")
    user_query = st.text_input("Your question", "")

    # Process uploaded files
    if uploaded_files:
        st.sidebar.write(f"Uploaded {len(uploaded_files)} file(s):")
        for file in uploaded_files:
            st.sidebar.write(f"- {file.name} ({file.size/1024:.1f} KB)")

    # Sidebar button to trigger ingestion of uploaded files
    if st.sidebar.button("Process Uploaded PDFs"):
        if not uploaded_files:
            st.sidebar.error("Please upload at least one PDF file first.")
        else:
            with st.spinner(f"Processing {len(uploaded_files)} PDF file(s)..."):
                # Create a temporary directory to store uploaded files
                temp_dir = tempfile.mkdtemp()
                try:
                    saved_files = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(file_path)
                    
                    st.info(f"Processing {len(saved_files)} PDF file(s)...")
                    
                    # Run ingestion with the temp directory
                    run_ingestion(temp_dir)                    

                    st.success("PDFs processed and indexed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
                finally:
                    # Clean up temporary files
                    shutil.rmtree(temp_dir)

    # Create or load the retrieval chain
    
    chain = build_retrieval_chain(openai_api_key=openai_api_key)

    if st.button("Submit Query"):
        if user_query.strip():
            with st.spinner("Generating answer..."):
                response = chain({"query": user_query})
            st.write("**Answer:**", response["result"])

            # Optional: display the source documents
            with st.expander("Show Source Documents"):
                for i, doc in enumerate(response["source_documents"], start=1):
                    st.markdown(f"**Source {i}**")

                    # Create a modified metadata dict with just the filename
                    display_metadata = doc.metadata.copy()
                    if "source" in display_metadata:
                        # Extract just the filename from the path
                        display_metadata["source"] = os.path.basename(display_metadata["source"])
        
                    st.write(display_metadata)
                    st.write(doc.page_content[:300], "...")
        else:
            st.warning("Please enter a question before submitting.")


if __name__ == "__main__":
    main()