from pymongo.mongo_client import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from bson import json_util
import json

def main():

    load_dotenv()
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGO_CONNECTION_STRING")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

    DB_NAME = "search_db"
    COLLECTION_NAME = "search_col"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vsearch_idx"

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", 
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        disallowed_special=(),
    )

    
    query = "is Qualcomm plotting a return"
    encoded_query = embeddings.embed_query(query)
    print(len(encoded_query))

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vsearch_idx",
                "path": "embedding",
                "queryVector": encoded_query,
                "numCandidates": 20,
                "limit": 5
                #"filter" : {"entity_name":{"$eq" : "Microsoft Corporation"}}
             }
        }
    ]

    search_col = client[DB_NAME][COLLECTION_NAME]
    docs = list(search_col.aggregate(pipeline))
    #print(docs)
    json_result = json_util.dumps({'docs': docs}, json_options=json_util.RELAXED_JSON_OPTIONS)
    print(json_result)
    

if __name__ == "__main__":
    main()
