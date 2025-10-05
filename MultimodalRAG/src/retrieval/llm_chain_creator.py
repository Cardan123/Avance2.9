from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from loguru import logger
try:  # Compatibilidad con Runnable
    from langchain_core.runnables import RunnableSerializable  # type: ignore
except Exception:  # pragma: no cover
    RunnableSerializable = object  # type: ignore

class LLMChainCreator:
    """
    Class to create an instance of LLMChain using a prompt_template and a language model (llm).
    """

    def __init__(self, prompt_template: PromptTemplate, llm):
        """
        Initialize the prompt template class and LLM class with a prompt_template and an llm.
        Args:
            prompt_template (PromptTemplate): The prompt template to use.
            llm: The language model to use.

        Returns:
            LLMChain: The created language chain instance.
        """

        self.prompt_template = prompt_template
        self.llm = llm
        self.create_chain()

    def create_chain(self):
        logger.info("[LLMChainCreator] Creating LLMChain with provided prompt template and LLM...")
        
        if not isinstance(self.prompt_template, PromptTemplate):
            raise ValueError("The provided prompt_template is not a valid PromptTemplate instance.")
        if not hasattr(self.llm, "invoke"):
            raise ValueError("The provided llm does not expose an 'invoke' method required by LLMChain.")
        # Loguear tipo para depurar adaptadores
        logger.debug(f"[LLMChainCreator] LLM type: {type(self.llm)}")
        
        # Intentar crear la cadena con una clave de salida personalizada 'answer'.
        # Algunas versiones antiguas de LangChain podrían no soportar 'output_key'.
        try:
            self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.llm, output_key="answer")
            logger.info("[LLMChainCreator] LLMChain created successfully with output_key='answer'.")
        except TypeError:
            self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.llm)
            logger.warning("[LLMChainCreator] 'output_key' not supported; chain will return key 'text'.")

        return self.llm_chain

# Example usage
# prompt_template = PromptTemplate(template="Responde a la pregunta: {question}", input_variables=["question"])
# llm = SomeLLMModel()
# chain_creator = LLMChainCreator(prompt_template, llm)
# llm_chain = chain_creator.create_chain()