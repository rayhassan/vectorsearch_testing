import pymongo
import sys
from sentence_transformers import SentenceTransformer, util
#from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import os
from dotenv import load_dotenv


def main():

    load_dotenv()
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    client = pymongo.MongoClient(connection_string)

    db = client["search_db"]
    collection = db["search_col"]
    index_name = "vsearch_idx"

    #query = "Microsoft and Net Zero by 2050"
    query = "Microsoft Corporation"

    vectorStore = MongoDBAtlasVectorSearch(collection, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), index_name=index_name)

    docs = vectorStore.max_marginal_relevance_search(query, K=3 )

    for doc in docs:
        print(doc.page_content)

if __name__ == "__main__":
    main()
