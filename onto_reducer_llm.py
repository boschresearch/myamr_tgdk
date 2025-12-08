# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import pandas as pd
import rdflib
import logging
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from azure.identity import get_bearer_token_provider, DefaultAzureCredential, AzureCliCredential
from pathlib import Path
import openai
import re
from langchain_community.chat_models import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class OntoReducerLLM:
    def __init__(self, 
                 csv_file_path: str, 
                 ontology_file_path: str, 
                 datasetname: str,
                 folderID: str
                 ) -> str:

        self.csv_file_path = csv_file_path
        self.ontology_file_path = ontology_file_path
        self.datasetname = datasetname
        self.folderID = folderID # target folder to store the reduced ontology in
        self.data = None
        self.reduced_ontology = None
        self.llm = None  # Will be initialized in login_to_azure_llm()

    def load_csv_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Loading CSV data from: {self.csv_file_path}")
            self.data = pd.read_csv(self.csv_file_path)
            return self.data
        except Exception as e:
            logging.error(f"Failed to load CSV file: {e}")
            raise

    def load_ontology_file(self) -> rdflib.Graph:
        try:
            logging.info(f"Loading ontology from Turtle file: {self.ontology_file_path}")
            g = rdflib.Graph()
            g.parse(self.ontology_file_path, format='turtle')
            logging.info("Ontology loaded successfully.")
            return g
        except Exception as e:
            logging.error(f"Failed to load ontology file: {e}")
            raise

    def reduce_ontology_with_llm(self, 
                                 datasetname: str, 
                                 input: str, 
                                 ontology: rdflib.Graph, 
                                 folderID: str
                                 ) -> str:

        self.folderID = folderID

        if not self.llm:
            self.login_to_azure_llm()

        ontology_text = ontology.serialize(format='turtle')
        ontology_text = ontology_text.replace('    ', '') 
        logging.info("Ontology serialized and minimized for token optimization.")

        prompt = f"Reduce the ontology based on the data such that only relevant elements remain in the reduced ontology that are needed for a mapping from the data to a Knowledge Graph with this Ontology. Note that the column names of the csv might not be literally the same as in the given ontology, although they can have the same meaning. The ontology has the correct literal terms. Note also that the reduced ontology must be a subset of the given ontology, you must not change any classes or relations. Return the ontology in ttl syntax only. Here is the Input:\n\nData:\n{input}\n\nOntology:\n{ontology_text}. "
        
        prompt_template = PromptTemplate(template=prompt, input_variables=["input", "ontology"])
        
        output_parser = StrOutputParser()

        chain = prompt_template | self.llm | output_parser

        result = chain.invoke({
            "data": input,
            "ontology": ontology_text
        })

        parsed_result = output_parser.parse(result)

        parsed_result = self.extract_ont(parsed_result)
        parsed_result = self.extract_more_ont(parsed_result)
        parsed_result = self.extract_more_more_ont(parsed_result)

        reduced_ontology_file_path = f"mapping-generation/generated_files/{self.datasetname}/{folderID}/reduced_ontology.ttl"
        with open(reduced_ontology_file_path, 'w') as file:
            file.write(parsed_result)

        logging.info("Reduced ontology successfully saved.")
        return parsed_result


    def extract_ont(self, text: str) -> str:
        # Define the regex pattern to match the content between ```ttl and ```
        pattern = re.compile(r'```ttl\s*(.*?)\s*```', re.DOTALL)
        match = pattern.search(text)

        if match:
            return match.group(1).strip()
        else:
            return text
        
    def extract_more_ont(self, text: str) -> str:
        # Define the regex pattern to match the content between ```turtle and ```
        pattern_more = re.compile(r'```turtle\s*(.*?)\s*```', re.DOTALL)
        match_more = pattern_more.search(text)

        if match_more:
            return match_more.group(1).strip()
        else:
            return text

    def extract_more_more_ont(self, text: str) -> str:
        # Define the regex pattern to match the content between ``` and ```
        pattern_more = re.compile(r'```\s*(.*?)\s*```', re.DOTALL)
        match_more = pattern_more.search(text)

        if match_more:
            return match_more.group(1).strip()
        else:
            return text

    def login_to_azure_llm(self) -> AzureChatOpenAI:
        """
        Log in to Azure LLM service using configuration from .env file and return the LLM object. And use LangChain.
        """
        try:
            logging.info("Loading configuration from .env file.")
            load_dotenv()

            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")

            if not all([azure_endpoint, api_version, deployment_name, model_name]):
                raise ValueError("Missing required configuration in .env file.")

            logging.info("Logging in to Azure OpenAI LLM service using .env configuration.")
            
            credential = AzureCliCredential()
            access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
            
            openai.azure_endpoint = azure_endpoint
            openai.api_version = api_version
            openai.azure_ad_token = access_token.token
            
            logging.info("Initializing the AzureChatOpenAI object.")
            self.llm = AzureChatOpenAI(
                deployment_name=deployment_name,
                model_name=model_name,
                azure_ad_token=openai.azure_ad_token,
                azure_endpoint=openai.azure_endpoint,
                openai_api_version=openai.api_version,
                temperature=0.6
            )
            
            logging.info("Successfully logged in to Azure LLM using .env configuration.")
            return self.llm
        except Exception as e:
            logging.error(f"Failed to log in to Azure LLM using .env configuration: {e}")
            raise


    def run(self) -> str:
        try:    
            input = self.load_csv_data()
            ontology = self.load_ontology_file()
            datasetname = self.datasetname
            self.reduced_ontology = self.reduce_ontology_with_llm(datasetname, 
                                                                  input, 
                                                                  ontology, 
                                                                  self.folderID)

            return self.reduced_ontology

        except Exception as e:
            logging.error(f"Error in execution: {e}")
            raise
