from retrieval.vector_search_pipeline import VectorSearchPipeline
from utils.model_manager import GeminiLLMWrapper
from retrieval.prompt_template_creator import PromptTemplateCreator
from retrieval.llm_chain_creator import LLMChainCreator
from retrieval.llm_answer_generation import LLMAnswerGeneration

class Retriever:
    def __init__(self):
        self.vector_search_pipeline = VectorSearchPipeline()
        self.gemini_llm_wrapper = GeminiLLMWrapper()
        self.prompt_template_wrapper = PromptTemplateCreator()
        self.llm_chain_creator = LLMChainCreator(self.prompt_template_wrapper.prompt_template, self.gemini_llm_wrapper.llm)
        self.answer_generator = LLMAnswerGeneration(self.llm_chain_creator.llm_chain)
        self.retriever_context = None

    def run(self):
        # Execute the vector search pipeline
        question = "¿Quien trabajo en la obra PROYECTO MULTIFAMILIAR JULIO C. TELLO?" 
        self.retriever_context = self.vector_search_pipeline.execute_vector_search(question)
        answer = self.answer_generator.generate_answer_with_llm_chain(self.retriever_context, question)
        return answer