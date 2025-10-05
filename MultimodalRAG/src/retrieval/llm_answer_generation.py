from loguru import logger
from langchain.chains import LLMChain

class LLMAnswerGeneration:
    """
    Class to handle answer generation using an LLMChain.
    """

    def __init__(self, llm_chain):
        """
        Initialize with an LLMChain.

        Args:
            llm_chain (LLMChain): The language model chain to use for generating answers.
        """
        self.llm_chain = llm_chain

    def generate_answer_with_llm_chain(self, retriever_context, question):
        """
        Generate an answer using the LLMChain.

        Args:
            retriever_context (str): The context retrieved by the retriever.
            question (str): The question to answer.

        Returns:
            str: The generated answer.
        """

        logger.info("[LLMAnswerGeneration] Generating answer with LLMChain...")

        if not retriever_context:
            logger.warning("[LLMAnswerGeneration] Retriever context is empty. Cannot generate answer.")
            return "No context available to generate an answer."
        if not question:
            logger.warning("[LLMAnswerGeneration] Question is empty. Cannot generate answer.")
            return "No question provided to generate an answer."

        logger.info(f"[magenta][LLMAnswerGeneration] Sending retriever context... {retriever_context[:50]}...[/magenta]")  # Log the first 50 characters of context
        logger.info(f"[yellow][LLMAnswerGeneration] Question: {question} [/yellow]")

        response = self.llm_chain.run(document=retriever_context, question=question)

        cleaned_response = response.strip() if isinstance(response, str) else "Invalid response format."
        logger.info(f"[orange1][LLMAnswerGeneration] Cleaned response: {cleaned_response}[/orange1]")

        return f"{cleaned_response}\n"