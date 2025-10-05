import os
import json
from langchain.schema import Document
from utils.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
from utils.model_manager import RoBERTaModelConfiguration
from utils.model_manager import BGEModelConfiguration
from utils.model_manager import BGETextEmbedder
from utils.config_file_manager import ConfigFileManager
import yaml
from loguru import logger
import torch
from sklearn.decomposition import PCA

class VectorSearchConfiguration:
    """
    Encapsulate the loading and access to the Vector Search configuration,
    including config_retrieval.yaml and vector_pipeline_config.json.
    """
    def __init__(self):

        self.mongo = MongoClient()
        # Load config_retrieval.yaml
        self.retrieval_config_yaml_path = ConfigFileManager.default_yaml_path()
        self.retrieval_config = ConfigFileManager.load_yaml_config(self.retrieval_config_yaml_path)

        # Load vector_pipeline_config.json
        self.vector_search_json_path = self.retrieval_config.get(
                                        "vector_search_file_path",
                                        os.path.join(os.path.dirname(__file__), "vector_searches", "vector_pipeline_config.json"))
        self.vector_search_index_json_path = self.retrieval_config.get(
                                        "vector_search_index_file_path",
                                        os.path.join(os.path.dirname(__file__), "vector_searches", "vector_search_index.json"))

        self.vector_search_json_template = ConfigFileManager.load_vector_search_template_json(self.vector_search_json_path)
        self.vector_search_index_json_template = ConfigFileManager.load_vector_search_template_json(self.vector_search_index_json_path)
        self.vector_search_dimension = self.retrieval_config.get("vector_search_dimension", 8)
        self.vector_search_apply_pca = self.retrieval_config.get("vector_search_apply_pca", False)
        self.vector_search_add_vector = self.retrieval_config.get("vector_search_add_vector", False) 
        self.vector_search_main_keyword = self.retrieval_config.get("vector_search_index_main_keyword", "text")
        self.vector_search_main_image_keyword = self.retrieval_config.get("vector_search_index_main_image_keyword", "text")
        self.vector_query_log_value = self.retrieval_config.get("log_vector_query_value", False)
        self.image_verbalization_use = self.retrieval_config.get("image_verbalization_use", True)
        self.checker_enabled = self.retrieval_config.get("checker_enabled", False)
        self.reranking_enabled = self.retrieval_config.get("reranking_enabled", True)
        self.reranking_model = self.retrieval_config.get("reranking_model", "rerank-english-v3.0")
        
        #self.vector_search_index_name = self.retrieval_config.get("vector_search_index_name", "vector_search_index")
        #self.vector_search_top_k = self.retrieval_config.get("vector_search_index_top_k", 10)
        #self.vector_search_exact = self.retrieval_config.get("vector_search_exact", True)
        #self.vector_search_embedding_path = self.retrieval_config.get("vector_search_embedding_path", "embedding")
        #self.vector_search_metric = self.retrieval_config.get("vector_search_metric", "cosine")

        
        

    def _default_yaml_path(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "config_retrieval.yaml")

    def _load_yaml_config(self, path):
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"[VectorSearchConfiguration] Loaded YAML config from {path}")
            return config
        except Exception as e:
            logger.error(f"[VectorSearchConfiguration] Error loading YAML config from {path}: {e}")
            return {}

    def _load_vector_search_template_json(self, path):
        try:
            with open(path, "r") as f:
                pipeline = f.read()
            logger.info(f"[VectorSearchConfiguration] Loaded vector search template from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"[VectorSearchConfiguration] Error loading vector search template from {path}: {e}")
            return ""

class VectorSearchPipeline:
    def __init__(self, config_path=None):
        logger.info("[VectorSearchPipeline] Initializing VectorSearchPipeline...")
        self.embeddings_model = BGEModelConfiguration()
        self.vector_search_config = VectorSearchConfiguration()
        self.add_vector_search_index_from_file()
        logger.info("[VectorSearchPipeline] VectorSearchPipeline initialized.")
        
    
    def load_pipeline_with_placeholders(self, variables):
        # Leer el archivo como texto plano para permitir placeholders sin comillas
        pipeline_str = self.vector_search_config.vector_search_json_template
    
        apply_variables = True

        if apply_variables:
            for key, value in variables.items():
                placeholder = f"${{{key}}}"
                # Para listas o valores complejos, serializa a JSON
                if isinstance(value, bool):
                    value_str = "true" if value else "false"
                elif isinstance(value, (list, dict, int, float)):
                    value_str = json.dumps(value)
                else:
                    # Usar json.dumps para strings para evitar comillas dobles duplicadas
                    value_str = json.dumps(value)
                pipeline_str = pipeline_str.replace(placeholder, value_str)
                
        logger.info("[VectorSearchPipeline] Loaded vector search pipeline with placeholders.")
        
        return json.loads(pipeline_str)
    
    def execute_vector_search(self, message = None):
        """"
        Execute the vector search using the configured pipeline and return results as Document objects.
            Args:
            message (str): The input text to be converted into embeddings for the vector search.
            queryVector (list): The vector to search for similar embeddings.
        Returns:
            str: A concatenated string of the list of documents objects containing the search results.
        """
        #test_vector = self.text_to_embeddings("Empty test vector")

        # 1. Encode the input text to get the query vector
        if message:
            #queryVector = self.embeddings_model.encode(message)[0] 
            queryVector = self.text_to_embeddings(message)

        if not message: queryVector = [0.24, 0.65, -0.10, 0.92, 0.40, -0.50, 0.59, -0.02]   

        if not queryVector:
            logger.error("[VectorSearchPipeline] No query vector provided or generated. Aborting vector search.")
            return ""
        
        # 2. Prepare the vector search using the configured pipeline
        logger.info("[VectorSearchPipeline] Executing vector search...")
        variables = {
            #"exact": self.vector_search_config.vector_search_exact,
            #"index": self.vector_search_config.vector_search_index_name,
            #"limit": self.vector_search_config.vector_search_top_k,
            #"path": self.vector_search_config.vector_search_embedding_path,
            "queryVector": queryVector
        }
        # 3. Load and fill the pipeline with actual values in place of placeholders
        if self.vector_search_config.vector_query_log_value:
            logger.info("[VectorSearchPipeline] Loading and filling the pipeline with variables: {}", variables)
        pipeline = self.load_pipeline_with_placeholders(variables)

        # 4. Execute the aggregation pipeline or called as Vector Search
        logger.info("[VectorSearchPipeline] Running aggregation pipeline on MongoDB collection...")
        results = self.vector_search_config.mongo.collection.aggregate(pipeline)
        results_list = list(results)
        logger.info("[VectorSearchPipeline] Retrieved {} results.", len(results_list))

        # 5. Convert results to Document objects
        documents = self._convert_results_to_documents(results_list)

        # 6. Concatenate page content from documents
        context_retriever = self.concatenate_page_content(documents)
        
        return context_retriever, documents
    
    def _convert_results_to_documents(self, results):
        """
        Convierte los resultados del vector search en una lista de objetos Document.
        """
        logger.info("[VectorSearchPipeline] Converting results to Document objects...")
        documents = []
        for result in results:
            markdown_context = result.get(self.vector_search_config.vector_search_main_keyword, "")
            image_content = result.get(self.vector_search_config.vector_search_main_image_keyword, "")
            main_context = markdown_context
            main_context_keyword = self.vector_search_config.vector_search_main_keyword
            #verifica si markdown_context existe en result, si existe usala como main_context
            if image_content:
                main_context = image_content
                main_context_keyword = self.vector_search_config.vector_search_main_image_keyword
            document = Document(
                page_content=main_context,  # Usa el campo principal definido en la configuración
                metadata={key: value for key, value in result.items() if key != main_context_keyword}  # Excluye el campo principal de la metadata
            )
            documents.append(document)
        logger.info("[VectorSearchPipeline] Converted {} results to Document objects.", len(documents))
        return documents
    
    def add_vector_search_index_from_file(self):
        """
        Agrega el índice vectorial a la colección si no existe, cargando la definición desde un archivo JSON.
        """
        logger.info("[VectorSearchPipeline] Adding vector search index to the collection...")

        try:
            # Leer la definición del índice desde el archivo JSON
            if self.vector_search_config.vector_search_add_vector:
                logger.info("[VectorSearchPipeline] Loading vector search index definition from file...")
                index_definition = json.loads(self.vector_search_config.vector_search_index_json_template)
                logger.info("[VectorSearchPipeline] Loaded index definition: \n{}", json.dumps(index_definition, ensure_ascii=False, indent=2))
                
                # Crear el modelo del índice
                search_index_model = SearchIndexModel(
                    definition=index_definition,
                    name=index_definition.get("name", "vector_search_index"),
                    type=index_definition.get("type", "vectorSearch"),
                )

                # Crear el índice en la colección
                if self.vector_search_config.vector_search_add_vector:
                    self.vector_search_config.mongo.collection.create_search_index(search_index_model)
                    logger.info("[VectorSearchPipeline] Vector search index '{}' created successfully.",
                                index_definition.get("name", "vector_search_index"))
            else:
                logger.info("[VectorSearchPipeline] Vector search index creation skipped as per configuration.")
        except Exception as e:
            logger.error("[VectorSearchPipeline] Failed to create vector search index: {}", e)

    def add_vector_search_index_from_config(self):
        """
        Agrega el índice vectorial a la colección si no existe.
        """
        logger.info("[VectorSearchPipeline] Adding vector search index to the collection...")

        # Definir el modelo del índice
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": self.vector_search_config.vector_search_embedding_path,
                        "numDimensions": self.vector_search_config.vector_search_dimension,
                        "similarity": self.vector_search_config.vector_search_metric
                    }
                ]
            },
            name=self.vector_search_config.vector_search_index_name,
            type="vectorSearch",
        )

        # Crear el índice en la colección
        try:
            if self.vector_search_config.vector_search_add_vector:
                self.vector_search_config.mongo.collection.create_search_index(search_index_model)
                logger.info("[VectorSearchPipeline] Vector search index '{}' created successfully.", 
                        self.vector_search_config.vector_search_index_name)
            else:
                logger.info("[VectorSearchPipeline] Vector search index creation skipped as per configuration.");   
        except Exception as e:
            logger.error("[VectorSearchPipeline] Failed to create vector search index: {}", e)

    def concatenate_page_content(self, documents):
        """
        Itera sobre todos los documentos y concatena el contenido de la página (page_content).+
        Iterate over all documents and concatenate the page content (page_content).

        Args:
            documents (list): List of documents.

        Returns:
            str: CConcatenated string of all page contents.
        """
        concatenated_content = " ".join(doc.page_content for doc in documents if doc.page_content)
        logger.info("[VectorSearchPipeline] Concatenated page content from {} documents.", len(documents))
        return concatenated_content
    
    def text_to_embeddings(self, text):
        """
        Convert a given text into embeddings using the loaded RoBERTa model.

        Args:
            text (str): The input text to convert into embeddings.

        Returns:
            list: A vector representing the embeddings of the input text.
        """
        logger.info("[VectorSearchPipeline] Converting text to embeddings using model: {}...".format(self.embeddings_model.model_name))
        try:
            # Tokenize the input text
            inputs = self.embeddings_model.tokenizer(text, return_tensors="pt")

            # Generate embeddings using the model
            with torch.no_grad():
                outputs = self.embeddings_model.model(**inputs)

            # Extract the embeddings (e.g., from the last hidden state)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

            # Reduce dimensions to 8 using PCA if enabled in the configuration
            if self.vector_search_config.vector_search_apply_pca:
                embeddings_dim = len(embeddings)
                logger.info("[VectorSearchPipeline] Original embedding dimensions: {}".format(embeddings_dim))
                if len(embeddings) < self.vector_search_config.vector_search_dimension:
                    # Directly truncate to the required dimensions if PCA is not applicable
                    reduced_embeddings = embeddings[:self.vector_search_config.vector_search_dimension]
                    logger.info("[VectorSearchPipeline] Truncated embedding dimensions to {}.".format(self.vector_search_config.vector_search_dimension))
                else:
                    # Check if PCA is applicable
                    if embeddings_dim < self.vector_search_config.vector_search_dimension:
                        logger.warning("[VectorSearchPipeline] PCA cannot be applied due to insufficient samples. Truncating instead.")
                        reduced_embeddings = embeddings[:self.vector_search_config.vector_search_dimension]
                    else:
                        # Reduce dimensions to the required size using PCA
                        pca = PCA(n_components=self.vector_search_config.vector_search_dimension)
                        reduced_embeddings = pca.fit_transform(embeddings)[0]
                        logger.info("[VectorSearchPipeline] Successfully reduced embedding dimensions to {} using PCA.".format(self.vector_search_config.vector_search_dimension))
                return reduced_embeddings
            else:
                return embeddings
        except Exception as e:
            logger.error("[VectorSearchPipeline] Failed to convert text to embeddings: {}", e)
            raise