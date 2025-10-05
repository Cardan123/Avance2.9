from langchain.prompts import PromptTemplate
from utils.config_file_manager import ConfigFileManager
from loguru import logger

class PromptTemplateCreator:
    """
    Class to create prompt templates for the language model.
    """

    def __init__(self):
        """
        Initializes the class with a template and input variables.

        Args:
            template (str): The prompt template.
            input_variables (list): List of input variables for the template.
        """
        self.config = ConfigFileManager.load_yaml_config(ConfigFileManager.default_yaml_path())
        self.json_template = self.load_from_json_file(self.config.get("prompt_template_path", ""))
        #self.template = self.json_template.get("template", "")
        #self.input_variables = self.json_template.get("input_variables", [])
        self.create_generic_prompt_template()


    def create_generic_prompt_template(self) -> PromptTemplate:
        """
        Creates and returns a PromptTemplate object.

        Returns:
            PromptTemplate: The created prompt template.
        """
        #logger.info("[PromptTemplateCreator] Creating PromptTemplate with template: {} and input_variables: {}", self.template, self.input_variables)
        self.prompt_template = PromptTemplate(template=self.template, input_variables=self.input_variables)  
        logger.info("[PromptTemplateCreator] PromptTemplate created successfully.")
        return self.prompt_template


    def load_from_json_file(self, file_path: str):
        """
        Loads the template and input variables from a JSON file in plain text format.

        Args:
            file_path (str): Path to the JSON file containing the template and input variables.

        Returns:
            None
        """
        import json
        try:
            logger.info("[PromptTemplateCreator] Loading prompt template from JSON file plain text: {}", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.template = data.get("template", "")
            self.input_variables = data.get("input_variables", [])
            logger.info("[PromptTemplateCreator] Template and input variables loaded successfully from file.")
        except FileNotFoundError:
            logger.error("[PromptTemplateCreator] JSON file not found: {}", file_path)
            raise FileNotFoundError("The provided JSON file does not exist.")
        except json.JSONDecodeError as e:
            logger.error("[PromptTemplateCreator] Error loading JSON from file: {}", e)
            raise ValueError("The provided JSON file is not valid.")

# Example usage
# template = "Answer the question: {question}"
# input_variables = ["question"]
# prompt_creator = PromptTemplateCreator(template, input_variables)
# prompt = prompt_creator.create_prompt_template()