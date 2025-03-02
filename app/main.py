import os
import sys
import streamlit as st

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ingest_pipeline import main as run_ingestion
from retrieval_chain import build_retrieval_chain

from dotenv import load_dotenv
load_dotenv()


def main():
    st.title("RAG Chatbot for Literature Review")

    # Sidebar button to trigger ingestion
    if st.sidebar.button("Run Ingestion Pipeline"):
        st.info("Running ingestion pipeline...this may take a few minutes.")
        run_ingestion()
        st.success("Ingestion completed!")

    # Create or load the retrieval chain
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    chain = build_retrieval_chain(openai_api_key=openai_api_key)

    # User query input
    st.subheader("Ask a question about the ingested PDFs")
    user_query = st.text_input("Your question", "")

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
