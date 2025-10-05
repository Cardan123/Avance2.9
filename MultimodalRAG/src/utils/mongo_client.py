from pymongo import MongoClient as PyMongoClient
import os
from loguru import logger
from dotenv import load_dotenv

class MongoClient:
    """
    Class for MongoDB Atlas operations.
    """
    def __init__(self):
        # load environment variables from .env file
        load_dotenv()
        uri = os.getenv("MONGODB_ATLAS_URI")
        self.client = PyMongoClient(uri)
        self.db = self.client.get_database(os.getenv("MONGODB_ATLAS_DB", "MultimodalRAG"))
        self.collection_name = os.getenv("MONGODB_ATLAS_COLLECTION", "multimodal-rag")
        self.collection = self.db.get_collection(os.getenv("MONGODB_ATLAS_COLLECTION", "multimodal-rag"))
        self.connected = self._check_connection()
        self._verify_database_exists()
        self._verify_collection_exists()
        self._verify_vector_index_exists()

    def insert_embedding(self, embedding_dict):
        """
        Inserts an embedding document into MongoDB Atlas.
        """
        self._verify_database_exists()
        self._verify_collection_exists()
        logger.info("[MongoClient] Inserting embedding document into MongoDB...")
        self.collection.insert_one(embedding_dict)

    def _verify_database_exists(self):
        """Check if the database exists."""
        logger.info("[MongoClient] Checking database existence...")
        db_names = self.client.list_database_names()
        if self.db.name not in db_names:
            logger.warning(f"[MongoClient] Database '{self.db.name}' does not exist. It will be created upon inserting the first document.")
        else:
            logger.info(f"[MongoClient] Database '{self.db.name}' exists.")

    def _verify_collection_exists(self):
        """Check if the collection exists."""
        logger.info("[MongoClient] Checking collection existence...")
        collection_names = self.db.list_collection_names()
        if self.collection.name not in collection_names:
            logger.warning(f"[MongoClient] Collection '{self.collection.name}' does not exist. It will be created upon inserting the first document.")
        else:
            logger.info(f"[MongoClient] Collection '{self.collection.name}' exists.")

    def _verify_vector_index_exists(self, index_name="vector_search_index"):
        """Check if the vector index exists."""
        logger.info("[MongoClient] Checking vector index existence...")
        indexes = self.collection.list_search_indexes()
        found = False
        for idx in indexes:
            # idx can be a dict with 'name' key
            if isinstance(idx, dict) and idx.get("name") == index_name:
                found = True
                break
        if not found:
            logger.warning(f"[MongoClient] Vector index '{index_name}' does not exist. Please create it for vector search functionality.")
        else:
            logger.info(f"[MongoClient] Vector index '{index_name}' exists.")

    def _check_connection(self):
        try:
            # The ismaster command is cheap and does not require auth.
            logger.info("[MongoClient] Checking MongoDB connection...")
            self.client.admin.command({'ping':1})
            logger.info("[MongoClient] MongoDB connection successful.")
            return True
        except Exception as e:
            logger.error(f"[MongoClient] MongoDB connection error: {e}")
            return False
        

""" if __name__ == "__main__":
    mongo_client = MongoClient()
    if mongo_client.connected:
        logger.info("[MongoClient] MongoDB client initialized and connected successfully.")
    else:
        logger.error("[MongoClient] Failed to connect to MongoDB.") """