from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from retrieval import retriever  # exported via retrieval/__init__.py

class GenerateChain:
    """
    Encapsula la construcción del prompt y la llamada al LLM.
    Opcionalmente puede usar el retriever si decides recuperar dentro (no solo pasar contexto).
    """
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        prompt: Optional[ChatPromptTemplate] = None,
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.parser = StrOutputParser()
        self.max_context_chars = 900000  # Ajusta según el modelo y necesidades
        # Precompone runnable (prompt | llm | parser)
        self.runnable = self.prompt | self.llm | self.parser

    def _truncate_context(self, context: str) -> str:
        if len(context) > self.max_context_chars:
            return context[: self.max_context_chars] + "\n...[contexto truncado]..."
        return context

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Versión síncrona. inputs requiere: question, document (string).
        """
        question = inputs.get("question", "")
        document = inputs.get("document", "")

        # Posible recuperar aquí:
        # if not document and self.retriever:
        #     docs = self.retriever.get_relevant_documents(question)
        #     document = "\n\n".join(d.page_content for d in docs)

        document = self._truncate_context(document)
        answer = self.runnable.invoke({"question": question, "document": document})
        return {"answer": answer}

    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", "")
        document = self._truncate_context(inputs.get("document", ""))
        answer = await self.runnable.ainvoke({"question": question, "document": document})
        return {"answer": answer}