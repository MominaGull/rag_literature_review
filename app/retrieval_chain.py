from langchain.chains import RetrievalQA, TransformChain, LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

from ingestion.embedding import get_openai_embeddings
from vectorstore.pinecone_index import get_or_create_pinecone_index


def build_retrieval_chain(openai_api_key: str, 
                          model_name: str = "gpt-4o",top_k: int = 3):
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

    template = """You are an AI research assistant helping with questions about scientific papers on resistance spot welding.
    
    Use ONLY the following context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer this question." Don't make up information.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer in a comprehensive, scientific manner. Include relevant details from the context.
    """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    query_transform_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Given a user question, reformulate it to be a standalone question that will help retrieve relevant context from a vector database about resistance spot welding research.
        
        Original question: {question}
        
        Reformulated question:"""
    )

    query_transformer = LLMChain(
        llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0),
        prompt=query_transform_prompt,
        output_key="reformulated_question"
    )

    def transform_query(inputs):
        question = inputs["query"]
        transformed = query_transformer.run(question)
        return {"transformed_query": transformed}

    query_transform_chain = TransformChain(
        input_variables=["query"],
        output_variables=["transformed_query"],
        transform=transform_query
    )

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
    retriever = vectorstore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": top_k})

    # 5. Create the LLM (ChatOpenAI or standard OpenAI LLM)
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                     model_name=model_name, temperature=0)

    # 6. Build the RetrievalQA chain
    chain_with_custom_input = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        input_key="transformed_query"
    )

    final_chain = SequentialChain(
        chains=[query_transform_chain, chain_with_custom_input],
        input_variables=["query"],
        output_variables=["result", "source_documents"]
    )

    return final_chain
