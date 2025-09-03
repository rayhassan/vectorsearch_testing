from pymongo.mongo_client import MongoClient
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


def main():

    load_dotenv()
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGO_CONNECTION_STRING")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    DB_NAME = "search_db"
    COLLECTION_NAME = "search_col"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vsearch_idx"

    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", 
        openai_api_key = os.getenv("OPENAI_API_KEY"),
    )

    loader = MongodbLoader(
        connection_string=os.getenv("MONGO_CONNECTION_STRING"),
        db_name="search_db",
        collection_name="search_col",
        #filter_criteria={"borough": "Bronx", "cuisine": "Bakery"},
        field_names=["event_text"],
    )

    #loader = CSVLoader(file_path="/Users/ray.hassan/Downloads/export_test.csv")
    json_docs = loader.load()

    #print(len(json_docs))
    #print(json_docs[0].page_content)
    print()
    for i in range(len(json_docs)):
        event_text_embeddings = embeddings.embed_documents(json_docs[i].page_content) 
        #print(i, len(event_text_embeddings))
        #print(len(event_text_embeddings[0]))
        #print(event_text_embeddings[0])

    collection = client[DB_NAME][COLLECTION_NAME]
    for embedding in event_text_embeddings:
        collection.update_many(
            {}, {'$set': {'embedding': embedding}}, upsert=False)
    
   client.close() 

if __name__ == "__main__":
    main()
