# This script ingests preprocessed regulatory data from CSV, splits it into text chunks, embeds those chunks, and loads the resulting vectors into a Qdrant vector database.
#
# - Loads environment variables for API keys and Qdrant host configuration from a parent .env file.
# - Reads the structured CSV output from the HTML parser script, treating the 'Source' column as a document identifier.
# - Splits the text of each regulatory section into overlapping chunks (for retrieval-friendly context) using a tiktoken-based splitter.
# - Uses OpenAI embeddings to convert each chunk into a 1536-dimensional vector.
# - Connects to Qdrant, deletes any existing collection named "raven", then creates a new collection with cosine distance for similarity search.
# - Uploads all embedded chunks with metadata to Qdrant for fast semantic retrieval and future querying.

import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# SPLITTING AND EMBEDDING SETTINGS
chunk_size=2000
chunk_overlap=200

print("Loading environment variables...")
env_path = Path(__file__).parent.parent / ".env"
success = load_dotenv(dotenv_path=env_path, override=True)
print(f".env found and loaded? {success}")

openAI_key = os.environ["OPENAI_API_KEY"]
qdrant_key = os.environ["QDRANT_API_KEY"]
print("QDRANT_API_KEY =", repr(qdrant_key))
qdrant_host = os.environ["QDRANT_HOST"]
print("QDRANT_HOST =", repr(qdrant_host))
print("Environment variables loaded.")

print("Loading CSV data...")
loader = CSVLoader(
    file_path="./sources/temp/temporary_regulatory_data_ready.csv",
    source_column="Source",
)
print("CSV data loaded.")

print("Running text splitter...")
documents = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(documents)
print("Text splitting complete.")

print("Running embeddings...")
embeddings = OpenAIEmbeddings()
print("Embeddings complete.")

print("Initializing Qdrant client...")
url = qdrant_host
api_key = qdrant_key
qdrant_collection = "raven"
qdrant_client = QdrantClient(url=url, api_key=api_key)
print("Qdrant client initialized.")

print("Deleting existing Qdrant collection if it exists...")
qdrant_client.delete_collection(collection_name=qdrant_collection)
print("Collection deleted.")

print("Creating new Qdrant collection...")
qdrant_client.create_collection(
    collection_name=qdrant_collection,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
print("New collection created.")

print("Populating Qdrant collection...")
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name=qdrant_collection,
)
print("Qdrant collection populated.")

print("Script execution completed.")
