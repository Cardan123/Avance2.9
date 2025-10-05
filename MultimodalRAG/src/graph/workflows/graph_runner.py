import os
from graph.graph import WorkflowGraph
from loguru import logger
from pathlib import Path  
from typing import Any, Dict
# Removed MermaidDrawMethod import (not available in installed mermaid package)
# print current working directory


class WorkflowRunner:
    @staticmethod
    def run_workflow_graph(
        question: str,
        retriever,
        logger = logger
    ) -> dict[str, Any]:
        """Run the workflow graph.

        Parameters
        ----------
        question : str
            Input user question.
        retriever : Any
            Retriever instance used by the workflow graph.
        logger : Any, optional
            Logger to use (defaults to module logger).

        Returns
        -------
        dict[str, Any]
            The answer payload produced by the workflow. 
        """
        wf = WorkflowGraph(retriever)
        initial_state = wf.initialize_state(question)
        app = wf.compile()
        WorkflowRunner.save_graph_image(app, "wf_graph.png")
        workflow_graph = app.invoke(initial_state)
        logger.info("[Workflow Runner] Workflow graph keys {}", workflow_graph.keys())
        answer_payload = workflow_graph["answer_payload"]
        logger.info("[Workflow Runner] Workflow graph completed with an answer.")
        return answer_payload

    @staticmethod
    def save_graph_image(app, output_file_path: str):
        """Save the graph image to a file."""
        try:
            output_file_path = Path(__file__).parent / output_file_path
            app.get_graph().draw_mermaid_png(
                output_file_path=output_file_path,
                max_retries=5,
                retry_delay=2.0,
            )
            logger.info(f"[Workflow Runner] Saved workflow graph to {output_file_path}")
        except Exception as e:
            logger.error(f"[Workflow Runner] Error saving workflow graph: {e}")