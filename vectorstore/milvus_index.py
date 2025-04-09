import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from langchain_milvus import Zilliz

load_dotenv(override=True)


ZILLIZ_CLOUD_URI = os.environ.get("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.environ.get("ZILLIZ_CLOUD_TOKEN")


def get_or_create_milvus_collection(embeddings, collection_name="LitReviewCollection"):
    # Initialize Milvus client
    client = MilvusClient(
        uri=ZILLIZ_CLOUD_URI,
        token=ZILLIZ_CLOUD_TOKEN
    )
    
    # Check if collection exists
    if not client.has_collection(collection_name):

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # Define schema
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            schema=schema
        )
    
    # Initialize LangChain Zilliz vector store
    vector_store = Zilliz(
        embedding_function=embeddings,
        connection_args={
            "uri": ZILLIZ_CLOUD_URI,
            "token": ZILLIZ_CLOUD_TOKEN,
        },
        collection_name=collection_name,
        consistency_level="Strong"
    )

    return vector_store
