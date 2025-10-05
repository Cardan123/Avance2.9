import gradio as gr
from loguru import logger
from retrieval.retriever import Retriever
from agent_rag.agent_rag import AgentRAG
# Use package-relative import (running the app with "python -m src.main" or from the src folder)
# avoids needing to mutate sys.path and prevents ModuleNotFoundError when executing main.py directly.
from graph.workflows.graph_runner import WorkflowRunner
from PIL import Image
from typing import Union, List, Dict, Any
import os

class RAGChatInterface:
    def __init__(self, retriever: Retriever, agent_rag: AgentRAG = None):
        self.retriever = retriever
        self.agent_rag = agent_rag
        #if agent_rag is None:
            #self.agent_rag = AgentRAG(retriever)

    def naive_rag(self, user_input):
        # Step 1: Retrieve context using the retriever
        answer = "No answer generated."
        if self.retriever is not None:
            logger.info(f"[RAGChatInterface] Using retriever: {self.retriever}")
            retriever_context = self.retriever.vector_search_pipeline.execute_vector_search(message=user_input)
            answer = self.retriever.answer_generator.generate_answer_with_llm_chain(retriever_context, user_input)     
        return answer
    
    def agentic_rag(self, user_input):
        answer = "No answer generated."
        if self.agent_rag is not None:
            logger.info(f"[blue][RAGChatInterface] Using AgentRAG: {self.agent_rag}[/blue]")
            answer = self.agent_rag.run(user_input)
        return answer
    
    def llm_answer_with_images(self, answer_input: Union[str, Dict[str, Any]], n_images: int = 8):
        """Genera imágenes asociadas a la respuesta si se proporciona una carga útil con referencias.

        Soporta dos formatos de entrada:
        1. str  -> se considera directamente el texto de respuesta (sin referencias)
        2. dict -> debe contener las claves:
            - 'answer': str con el texto final
            - 'answer_references': List[str] con rutas (paths) a archivos de imagen

        Args:
            answer_input: Texto o payload con 'answer' y 'answer_references'.
            n_images: Límite máximo de imágenes a devolver (si hay más rutas, se truncan).

        Returns:
            (texto, lista_de_imagenes)
        """
        images: List[Image.Image] = []

        # Determinar el texto de respuesta y las rutas de referencia
        if isinstance(answer_input, dict):
            answer_text = str(answer_input.get('answer', ''))
            refs = answer_input.get('answer_references') or [] 
            if not isinstance(refs, (list, tuple)):
                logger.warning("[RAGChatInterface] 'answer_references' no es lista/tupla; se ignora.")
                refs = []
        else:
            answer_text = str(answer_input)
            refs = []

        # Truncar según n_images si aplica
        if n_images is not None and n_images > 0:
            refs = refs[:n_images]

        # Cargar imágenes de rutas
        for path in refs:
            try:
                if not isinstance(path, str):
                    logger.warning(f"[RAGChatInterface] Referencia no es string: {path} (type={type(path)})")
                    continue
                if not os.path.exists(path):
                    logger.warning(f"[RAGChatInterface] Ruta inexistente: {path}")
                    continue
                img = Image.open(path)
                images.append(img)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f"[RAGChatInterface] Error cargando imagen '{path}': {e}")

        # Si no se obtuvieron imágenes (sin refs o fallos), generar placeholders
        if not images:
            placeholder_count = n_images if (n_images and n_images > 0) else 1
            for _ in range(placeholder_count):
                images.append(Image.new('RGB', (160, 90), color='gray'))

        return answer_text, images

    def launch_interface(self):
        """
        Lanza una interfaz Gradio que devuelve tanto el texto de la respuesta
        como una imagen (image_test) usando la función llm_answer_with_image.
        """
        def chat_fn(message: str):
            logger.info(f"[blue][RAGChatInterface] Mensaje recibido: {message}[/blue]")
            # Elegimos explícitamente el flujo para mayor legibilidad
            if self.agent_rag is not None:
                logger.debug("[RAGChatInterface] Usando flujo Agentic RAG")
                answer_text = self.agentic_rag(message)
            else:
                logger.debug("[RAGChatInterface] Usando flujo Naive RAG")
                answer_text = self.naive_rag(message)

            final_text, images = self.llm_answer_with_images(answer_text, n_images=6)
            return final_text, images

        with gr.Blocks(theme="compact") as demo:
            gr.Markdown("# Multimodal RAG Chat Interface")
            gr.Markdown("Interfaz que muestra respuesta de RAG y múltiples imágenes generadas")
            with gr.Column():
                input_box = gr.Textbox(label="Pregunta", placeholder="Escribe tu mensaje...", lines=2)
                submit_btn = gr.Button("Enviar")
                # Opción 2: usamos Markdown para que la altura se ajuste automáticamente al contenido
                gr.Markdown("### Respuesta:")
                answer_box = gr.Markdown("")
                gallery = gr.Gallery(label="Imágenes Generadas", columns=3, height="auto")

            submit_btn.click(chat_fn, inputs=input_box, outputs=[answer_box, gallery])
            input_box.submit(chat_fn, inputs=input_box, outputs=[answer_box, gallery])
        # Habilita cola de Gradio para evitar ejecuciones concurrentes en recursos no thread-safe
        # default_concurrency_limit=1 asegura que sólo se procese una petición a la vez
        # max_size limita el tamaño de la cola para evitar acumulación excesiva
        demo.queue(max_size=32, default_concurrency_limit=1)
        demo.launch(share=True)