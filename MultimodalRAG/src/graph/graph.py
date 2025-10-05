from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from functools import partial
from loguru import logger
import os

from retrieval import retriever  # exported via retrieval/__init__.py
from graph.state import GraphState
from graph.consts import (
    RETRIEVE, GENERATE, IMAGE_CHECKER, IMAGE_PROCESSING, IMAGE_VERBALIZATION, CONTEXTING, 
    SPLITTER, IMAGE_CONTEXT, MARKDOWN_CONTEXT, RERANKING
)
from graph.nodes.retrieve_node import RetrieveNode
from graph.nodes.splitter_node import SplitterNode
from graph.nodes.reranking_node import RerankingNode
from graph.nodes.image_checker_node import ImageCheckerNode
from graph.nodes.image_processing_node import ImageProcessingNode
from graph.nodes.image_context_node import ImageContextNode
from graph.nodes.markdown_context_node import MarkdownContextNode
from graph.nodes.image_verbalization_node import ImageVerbalizationNode
from graph.nodes.generate_node import GenerateNode  
from graph.nodes.contexting_node import ContextingNode

load_dotenv()

class WorkflowGraph:
    def __init__(self, retriever):
        self.retriever = retriever
        self.workflow = StateGraph(GraphState)
        self.build_graph()

    def initialize_state(
            self, 
            question: str,
            documents: list[str] = None
        ) -> GraphState:

        """Initialize the graph state."""

        state = {
            "question": question,
            "documents": documents or [],
            "markdown_docs": [],
            "image_docs": [],
            "image_contexts": [],
            "image_paths": [],
            "image_descriptions": [],
            "image_descriptions_text": "",
            "context": "",
            "answer": "",
            "images_context": "",
            "markdowns_context": "",
            "markdowns_has_answers": False,
            "images_has_answers": False,
            "checker_context": "",
            "answer_image_paths": [],
            "answer_image_contents": [],
            "answer_image_justifications": [],
            "answer_payload": {},  # Inicializa el campo para el payload estructurado
        }

        return state

    def build_graph(self):
        """Build the graph."""
        self.workflow.add_node(RETRIEVE, partial(RetrieveNode.process, retriever=self.retriever))
        self.workflow.add_node(SPLITTER, SplitterNode.process)
        self.workflow.add_node(RERANKING, partial(RerankingNode.process, retriever=self.retriever))
        self.workflow.add_node(IMAGE_CHECKER, partial(ImageCheckerNode.process, retriever=self.retriever))
        self.workflow.add_node(IMAGE_PROCESSING, partial(ImageProcessingNode.process, retriever=self.retriever))
        self.workflow.add_node(IMAGE_VERBALIZATION, partial(ImageVerbalizationNode.process, retriever=self.retriever))
        self.workflow.add_node(IMAGE_CONTEXT, ImageContextNode.process)
        self.workflow.add_node(MARKDOWN_CONTEXT, MarkdownContextNode.process)
        self.workflow.add_node(GENERATE, partial(GenerateNode.generate, retriever=self.retriever))
        self.workflow.add_node(CONTEXTING, ContextingNode.process)

        # Define edges with conditions
        self.workflow.set_entry_point(RETRIEVE)
        self.workflow.add_edge(RETRIEVE, MARKDOWN_CONTEXT)
        self.workflow.add_edge(MARKDOWN_CONTEXT, SPLITTER)

        self.workflow.add_conditional_edges(
                SPLITTER,
                partial(SplitterNode.route_from_splitter_reranking, retriever=self.retriever),
                {
                    IMAGE_PROCESSING: IMAGE_PROCESSING,
                    CONTEXTING : CONTEXTING,
                    RERANKING: RERANKING,
                    END: END,  
                }
            )
        
        self.workflow.add_edge(RERANKING, IMAGE_PROCESSING)
        self.workflow.add_edge(IMAGE_PROCESSING, IMAGE_CONTEXT)
        
        # After processing images (paths), verbalize them before building context
        # Eliminado enlace directo MARKDOWN_CONTEXT -> CHECKER; el flujo pasa por SPLITTER
        self.workflow.add_conditional_edges(
                IMAGE_CONTEXT,
                partial(ImageContextNode.route_from_image_context, retriever=self.retriever),
                {
                    IMAGE_CHECKER: IMAGE_CHECKER,
                    CONTEXTING : CONTEXTING,
                    END: END,  
                }
            )
        self.workflow.add_edge(IMAGE_CHECKER, CONTEXTING)
        self.workflow.add_edge(CONTEXTING, GENERATE)
        self.workflow.add_edge(GENERATE, END)

    def compile(self):
        """Compile the graph."""
        return self.workflow.compile()