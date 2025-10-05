#from ingest.image_ingest_pipeline import ImageIngestPipeline
from retrieval.retriever import Retriever
from agent_rag.agent_rag import AgentRAG
from ui.rag_chat_interface import RAGChatInterface
import os
import sys
import root as project_root
from loguru import logger

from utils.logger_config import LoggerConfig


from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")
LAUNCH_UI = True

def main():
    try:
        LoggerConfig.setup_logger()
        logger.info("[Main] Application started successfully.")
        query = "Dame informacion sobre clases de concreto con una resistencia ala rotura de 210"
        #query = "dame informacion de Incluye información sobre el sistema estructural sismorresistente, periodo fundamental de vibración, parámetros de definición de fuerza sísmica"
        #query = "resistencia del concreto clase 1"
        # Initialize and run the image ingestion pipeline
        #ingestPipeline = ImageIngestPipeline()
        #ingestPipeline.run()

        # Initialize and run the retrieval pipeline
        retriever = Retriever()
        #retriever.run()
        # Initialize and run the AgentRAG
        agent_rag = AgentRAG(retriever)
        #agent_rag.run(query)

        # Initialize the RAG chat interface
        if LAUNCH_UI:
            rag_chat_interface = RAGChatInterface(retriever, agent_rag)
            #rag_chat_interface.agentic_rag(query)
            rag_chat_interface.launch_interface()

    except Exception as e:
        logger.error(f"[Main] An error occurred: {e}")

if __name__ == "__main__":
    main()