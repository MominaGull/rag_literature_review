from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

from ingestion.embedding import get_openai_embeddings
from vectorstore.pinecone_index import get_or_create_pinecone_index


def build_retrieval_chain(openai_api_key: str, 
                          model_name: str = "gpt-3.5-turbo",top_k: int = 3):
    """
    Builds a RetrievalQA chain that retrieves chunks from Pinecone
    and uses a ChatOpenAI model to generate an answer.
    
    Args:
        openai_api_key (str): OpenAI API key
        model_name (str, optional): OpenAI model name. Defaults to "gpt-3.5-turbo".
        top_k (int, optional): Number of chunks to retrieve from Pinecone. Defaults to 3.
        
    Returns:
        RetrievalQA: LangChain RetrievalQA chain object
    """

    # 1. Get or create the Pinecone Index
    index = get_or_create_pinecone_index()

    # 2. Create the embeddings object
    embeddings = get_openai_embeddings(api_key=openai_api_key)

    # 3. Build a VectorStore around that index
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"  # This should match how you stored chunk text
    )

    # 4. Turn the vectorstore into a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 5. Create the LLM (ChatOpenAI or standard OpenAI LLM)
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                     model_name=model_name, temperature=0)

    # 6. Build the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # So we can see which chunks were used
    )

    return chain
