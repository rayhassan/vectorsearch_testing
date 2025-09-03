import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pymongo
import pandas as pd
import os
from dotenv import load_dotenv

# Load MiniLM model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)

# Connect to MongoDB
load_dotenv()
connection_string = os.getenv("MONGO_CONNECTION_STRING")
client = pymongo.MongoClient(connection_string)
db = client['search_db']
collection = db['search_coll'] 

# Function to generate embeddings for each text using MiniLM model
def generate_embeddings(text):
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
    return embeddings.tolist()

# Read CSV file into a pandas DataFrame
df = pd.read_csv('/Users/ray.hassan/Downloads/export_test.csv')

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Generate embeddings for the text
    embeddings = generate_embeddings(row['text_column'])
    
    # Convert each line into a MongoDB document
    document = {
        'text': row['text_column'],
        'embeddings': embeddings  # Embeddings for the text
    }
    
    # Insert the document into MongoDB
    collection.insert_one(document)

# Close MongoDB connection
client.close()

