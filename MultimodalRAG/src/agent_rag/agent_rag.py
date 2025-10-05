from graph.workflows.graph_runner import WorkflowRunner
from retrieval.retriever import Retriever
from loguru import logger

class AgentRAG:
    def  __init__(self, retriever):
        self.retriever = retriever
        self.graph_runner = WorkflowRunner

    def run(self, user_input):
        # Placeholder for agentic RAG implementation
        logger.info("[AgentRAG] Running agentic RAG with workflow graph...")
        answer = WorkflowRunner.run_workflow_graph(
            question=user_input,
            retriever=self.retriever
        )
        logger.info(f"[AgentRAG] Agentic RAG answer: {answer}")
        logger.info("[AgentRAG] Agentic RAG completed.")

        return answer
