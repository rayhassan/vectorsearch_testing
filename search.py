from pymongo.mongo_client import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
import os

def main():

    MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://rayh:mongodb4u@sandbox.fgkzb.mongodb.net/?retryWrites=true&w=majority&appName=sandbox"
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    DB_NAME = "search_db"
    COLLECTION_NAME = "search_col"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vsearch_idx"

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", 
        openai_api_key = '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',
        disallowed_special=(),
    )

    
   # vector_search = MongoDBAtlasVectorSearch.from_documents(
   #     documents=json_docs,
   #     embedding=embeddings, 
   #     collection=COLLECTION_NAME,
   #     index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
   #     )

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        #MONGODB_ATLAS_CLUSTER_URI,
        "mongodb+srv://rayh:mongodb4u@sandbox.fgkzb.mongodb.net/?retryWrites=true&w=majority&appName=sandbox",
        DB_NAME + "." + COLLECTION_NAME,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )

    query = "Microsoft Corporation"
    encoded_query = embeddings.embed_query(query)
    print(encoded_query)
    print(len(encoded_query))
    results = vector_search.similarity_search(encoded_query)
    #results = vector_search.max_marginal_relevance_search(COLLECTION_NAME, encoded_query,vector_field='embedding', K=3)

    print(results[0].page_content)
    

if __name__ == "__main__":
    main()
