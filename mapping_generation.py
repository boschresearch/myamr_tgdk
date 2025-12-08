# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import pandas as pd
import logging
from rdflib import Graph, URIRef, Literal, RDF
from rdflib.namespace import RDFS, RDF, Namespace
from rdflib.serializer import Serializer
from io import StringIO
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from azure.identity import get_bearer_token_provider, DefaultAzureCredential, AzureCliCredential
from pathlib import Path
import openai
from ruamel.yaml import YAML
from datetime import datetime
import re
import csv
import yaml
from ruamel.yaml import YAML

from langchain_community.chat_models import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv

import os

from onto_reducer_llm import OntoReducerLLM
from onto_reducer_similar import OntoReducerSim
from onto_reducer_naive import OntoReducerNaive
from eval_rml_relaxed import RMLEvaluator
from eval_yarrrml_relaxed import YarrrmlEvaluator

from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from typing import Dict, List
import json
import openai
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LLMMappingGenerator:
    def __init__(self,
                 datasetname: str, 
                 mapping_language: str,
                 ont_reduce_variant: str, 
                 datasize: str,
                 shot_variant: str) -> None:
        """
        Initialize the mapping generator.
        
        """
        self.datasetname = datasetname # specifies the dataset for the method execution
        self.mapping_language = mapping_language # specifies in which language the mapping should be generated
        self.ont_reduce_variant = ont_reduce_variant # specifies which ontology reduction method to employ
        self.datasize = datasize # specifies whether to use full data source example or column names only
        self.shot_variant = shot_variant # specifies the variant of shots to enhance the prompt
        self.data = None # Will be initialized by result from method load_csv_data()
        self.reduced_ontology = None # Will be initialized by result from method reduce_ontology() (after invoking load_ontology_file())
        self.prompt_instructions = None # Will be initialized by result from method load_txt_from_file() which is invoked in build_mapping_template()
        self.llm = None  # Will be initialized in login_to_azure_llm()
        self.mapping_template = None # Will be initialized, if zero shot variant with TEMPLATE is chosen, in build_prompt_template_mapping()

        self.csv_file_path = f"mapping-generation/input/{self.datasetname}/prompt_{mapping_language}/data.csv"
        self.ontology_file_path = f"mapping-generation/input/{self.datasetname}/prompt_{mapping_language}/ontology.ttl"

        prompt_instructions_path = "" # file path for the content of the prompt instructions (which depend on the shot variant)
        if shot_variant == "template":
            prompt_instructions_path = f"mapping-generation/input/task_instructions/{self.mapping_language}/template.txt"
        elif shot_variant == "few":
            prompt_instructions_path = f"mapping-generation/input/task_instructions/{self.mapping_language}/1-few_shot.txt"
        elif shot_variant == "zero":
            # if zero shot, then the zero shot task instructions are given.
            prompt_instructions_path = f"mapping-generation/input/task_instructions/{self.mapping_language}/0_shot.txt"                       
        else :
            logging.info("The key for the prompt instructions is invalid.")  

        self.prompt_instructions_path = prompt_instructions_path
        logging.info(f"prompt_instructions_path is set to: {self.prompt_instructions_path}")

    def load_csv_data(self) -> pd.DataFrame:      
        """Load CSV data into a DataFrame."""
        try:
            logging.info(f"Loading CSV data from: {self.csv_file_path}")
            if self.datasize == "first20" :
                self.data = pd.read_csv(self.csv_file_path)
                self.data = self.data.head(20)
                logging.info(f"Column names plus first 20 rows of data set loaded.")
            elif self.datasize == "columns-only" :
                self.data = pd.read_csv(self.csv_file_path)
                self.data = self.data.head(0)             
                logging.info(f"Columns only loaded.")   
            elif self.datasize == "full" :
                self.data = pd.read_csv(self.csv_file_path)
                logging.info(f"Full data set loaded.")                    
            else :
                logging.info("The key for the datasize is invalid.")  
                return None

            data_doc_path = f"mapping-generation/generated_files/{self.datasetname}/{self.folderID}/data.csv"
            self.data.to_csv(data_doc_path, index=False)

            return self.data   

        except Exception as e:
            logging.error(f"Failed to load CSV file: {e}")
            raise

    def load_prompt_from_file(self) -> str:
        try:
            self.prompt_instructions_path = Path(self.prompt_instructions_path)
            with open(self.prompt_instructions_path, 'r') as file:
                self.prompt_instructions = file.read()
            instructions_doc_path = f"mapping-generation/generated_files/{self.datasetname}/{self.folderID}/instructions.txt"
            with open(instructions_doc_path, 'w') as file:
                file.write(self.prompt_instructions)
            
            return self.prompt_instructions
        except Exception as e:
            logging.error(f"Failed to load template from {self.prompt_instructions_path}: {e}")
            raise

    def load_txt_from_file(self, template_file_path: str) -> str:
        try:
            logging.info(f"Loading mapping template, ontology or example shots from: {template_file_path}")
            template_path = Path(template_file_path)
            with open(template_path, 'r') as file:
                example_shot_data = file.read()
            return example_shot_data
        except Exception as e:
            logging.error(f"Failed to load template from {template_file_path}: {e}")
            raise

    def reduce_ontology(self, 
                        csv_file: str, 
                        ontology_file: str,
                        folderID: str
                        ) -> str:

        self.csv_file = csv_file
        self.ontology_file = ontology_file
        self.datasetname = datasetname
        self.folderID = folderID

        if self.ont_reduce_variant == "llm" :
            # VARIANT - LLM-based
            onto_reducer = OntoReducerLLM(self.csv_file, self.ontology_file, self.datasetname, self.folderID)
            self.reduced_ontology = onto_reducer.run()

        elif self.ont_reduce_variant == "naive" : 
            # VARIANT - naive reduction (returns class, label (if exists), comment (if exists), and properties via query (no domain/range))
            logging.info("Reducing the ontology naively to specified elements (independent of data input).")
            reducer = OntoReducerNaive(self.csv_file, self.ontology_file, self.datasetname, self.folderID)
            self.reduced_ontology = reducer.run(self.datasetname, self.folderID, self.csv_file, self.ontology_file)

        elif self.ont_reduce_variant == "similar" :
            # VARIANT - Baseline, currently trying lexicographical matching with column names              
            logging.info("Reducing the ontology with embedding and similarity selection with given input data and ontology.")
            onto_reducer = OntoReducerSim(self.csv_file, self.ontology_file, self.datasetname, self.folderID)
            self.reduced_ontology = onto_reducer.run()
        
        return self.reduced_ontology

    def build_prompt_template_mapping(self) -> PromptTemplate:
        """
        Define and build the mapping template for the LLM for the zero shot with TEMPLATE variant.
        """

        template_file_path = f"mapping-generation/input/{self.datasetname}/prompt_{mapping_language}/0_mapping_template.txt"
        self.mapping_template = self.load_txt_from_file(template_file_path)
        logging.info(f"This mapping template is added to the prompt as enhancement: {self.mapping_template}")

        prompt_template = """ 
                {prompt_instructions}
                Data:
                {data}
                Ontology:
                {ontology}
                Mapping Template:
                {mapping_template}
                """

        prompt_full = PromptTemplate(
            input_variables=[
                "prompt_instructions",
                "data",
                "ontology",
                "mapping_template"
            ],
            template=prompt_template
        )
        
        logging.info(f"Prompt template built. It looks like this: {prompt_full}")
        return prompt_full

    def build_prompt_template_zero(self) -> PromptTemplate:
        """
        Define and build the mapping template for the LLM for the ZERO shot variant.
        """
        zero_template = """ 
                {prompt_instructions}
                Data:
                {data}
                Ontology:
                {ontology}
                """

        prompt_full = PromptTemplate(
            input_variables=[
                "prompt_instructions",
                "data",
                "ontology",
            ],
            template=zero_template
        )
        
        return prompt_full

    def login_to_azure_llm(self) -> AzureChatOpenAI:
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

    def get_few_shot_prompt(self, few_shot_examples, input_ontology, input_data):
        """
        Returns the most relevant few-shot examples with the provided prompt
        """
        to_vectorize = [" ".join(example.values()) for example in few_shot_examples]
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        vectorstore = FAISS.from_texts(to_vectorize, embeddings, metadatas=few_shot_examples)

        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=1  # we pick 1 (the most similar) example
        )

        examples = example_selector.select_examples({f"input_ontology":input_ontology, "input_data":input_data})

        few_shot_prompt = "Examples:\n"
        for i in examples:
            few_shot_prompt += f"""
                \nInput Ontology:
                {i["input_ontology"]}
                Input Data:
                {i["input_data"]}
                Result Mappings:
                {i["result_mappings"]}\n
            """
        
        return few_shot_prompt

    def get_few_shot_examples(self, json_file_path):
        try:
            with open(json_file_path, 'r') as file:
                examples = json.load(file)
                return examples
        except FileNotFoundError:
            print(f"Error: The file {json_file_path} was not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {json_file_path}.")
            return []

    def construct_prompt(self, best_few_shot_example):

        prompt_instructions = self.prompt_instructions
        
        final_prompt = PromptTemplate(
            input_variables=[
                "input_ontology",
                "input_data",
            ],
            template = prompt_instructions +
                """
                Ontology:
                {input_ontology}
                Data:
                {input_data}
                """ + best_few_shot_example
        )

        logging.info(f"This is the final prompt template including with instructions and few shot already included: {final_prompt}")

        return final_prompt

    def generate_mapping(
        self,
        prompt_full: PromptTemplate
    ) -> str:
        if not self.llm:
            self.login_to_azure_llm()

        data_for_prompt = self.data.to_csv(index=False, header=True)

        output_parser = StrOutputParser()

        logging.info("Starting LLM chain for mapping generation with/without a generic mapping template (but without a shot example).")
        
        debug_info = {
            "prompt_instructions": self.prompt_instructions,
            "data": data_for_prompt,
            "ontology": self.reduced_ontology,
            "mapping_template": self.mapping_template
        }

        chain = prompt_full | self.llm | output_parser

        print(prompt_full.template.format(**debug_info))

        result = chain.invoke(debug_info)

        return result

    def extract_mapping(self, 
                     mapping_result: str, 
                     mapping_language: str):
        
        if mapping_language == "yarrrml":
            # Define the regex pattern to match the content between ```yaml and ```
            pattern = re.compile(r'```yaml\s*(.*?)\s*```', re.DOTALL)
            match = pattern.search(mapping_result)

            if match:
                return match.group(1).strip()
            else:
                return mapping_result

        elif mapping_language == "rml":
            # Define the pattern to match content between prefix: ```rml, ```RML, ```ttl, ```TTL or ```turtle and suffix: ```
            pattern = re.compile(r'```(rml|RML|ttl|TTL|turtle)\s*(.*?)\s*```', re.DOTALL)
            match = pattern.search(mapping_result)

            if match:
                return match.group(2).strip()
            else:
                return mapping_result

    def add_unknown_prefixes(self, 
                             generated_mappings: str, 
                             mapping_language: str):

        if mapping_language == "rml":
            mapping_graph = Graph()
            try:
                mapping_graph.parse(data=generated_mappings, format="turtle")
            except Exception as e:
                print(f"Error loading RML/R2RML mappings: {e}")
                return None

            mapping_prefixes = dict(mapping_graph.namespaces())
            
            used_uris = set()
            for s, p, o in mapping_graph:
                if isinstance(s, URIRef):
                    used_uris.add(str(s))
                if isinstance(p, URIRef):
                    used_uris.add(str(p))
                if isinstance(o, URIRef):
                    used_uris.add(str(o))

            ontology_graph = Graph()
            try:
                ontology_graph.parse(data=self.reduced_ontology, format="turtle")
            except Exception as e:
                print(f"Error loading reduced ontology: {e}")
                return None

            ontology_prefixes = dict(ontology_graph.namespaces())

            new_prefixes = {}
            for uri in used_uris:
                for prefix, namespace in ontology_prefixes.items():
                    if str(namespace) in uri and prefix not in mapping_prefixes:
                        new_prefixes[prefix] = namespace
                        logging.info(f"The namespace {prefix}:{namespace} has been added to the generated mappings")

            if new_prefixes:
                prefix_declarations = "\n".join(f"@prefix {prefix}: <{namespace}> ." 
                                            for prefix, namespace in new_prefixes.items())
                
                prefix_section_end = re.search(r"((?:@prefix[^\n]+\n)+)", generated_mappings)
                if prefix_section_end:
                    insert_position = prefix_section_end.end()
                    updated_mappings = (generated_mappings[:insert_position] + 
                                    "\n" + prefix_declarations + "\n" +
                                    generated_mappings[insert_position:])
                else:
                    updated_mappings = prefix_declarations + "\n\n" + generated_mappings
                    
                return updated_mappings
            
            return generated_mappings

        elif mapping_language == "yarrrml":
            generated_mappings = generated_mappings.strip()
            
            if not generated_mappings:
                logging.info("Generated mappings input is empty or only whitespace.")
                return None

            yaml = YAML()
            yaml.preserve_quotes = True
            try:
                mappings = yaml.load(generated_mappings)
            except Exception as e:
                print(f"Error loading YARRRML mappings: {e}")
                return None

            if mappings is None:
                print("Mappings is None after loading. Please check the input.")
                return None

            if 'prefixes' not in mappings:
                print("The 'prefixes' key is missing from the mappings.")
                return None

            used_prefixes = set()
            for mapping_content in mappings.get('mappings', {}).values():
                s_value = mapping_content.get('s', '')
                if isinstance(s_value, str) and ":" in s_value:
                    used_prefixes.add(s_value.split(":", 1)[0])

                for po_entry in mapping_content.get('po', []):
                    for element in po_entry:
                        if isinstance(element, str) and ":" in element:
                            used_prefixes.add(element.split(":", 1)[0])

            specified_prefixes = mappings.get('prefixes', {}).keys()

            graph = Graph()
            graph.parse(data=self.reduced_ontology, format="turtle")
            ontology_prefix_dict = {}
            for prefix, namespace in graph.namespaces():
                ontology_prefix_dict[prefix] = namespace

            for prefix in used_prefixes:
                if prefix not in specified_prefixes:
                    if prefix in ontology_prefix_dict.keys():
                        mappings['prefixes'][str(prefix)] = str(ontology_prefix_dict[prefix])
                        logging.info(f"The namespace {str(prefix)}:{str(ontology_prefix_dict[prefix])} has been added to the generated mappings")
                    else:
                        logging.warning(f"Prefix not found in the provided: {prefix}. Hence cannot be added to generated mappings")
            
            output = StringIO()
            yaml.dump(mappings, output)

        return output.getvalue()

    def run(self, 
            folderID: str, 
            real_run: bool
            ) -> str:
        """
        Run the complete pipeline to generate a mapping: 
            (1) loading the data sample (including storing data into file for documentation purposes) and 
            (2) reducing the ontology (each ontology method loads the ontology on its own) and
            (3) loading the task instructions and 
            (4) either pulling the mapping template or preparing the best from the few shot examples
            (5) calling an LLM to generate a mapping
            (6) saving the mapping to file
            (7) evaluate the mapping against a reference mapping
        """
        try:
            self.folderID = folderID
            self.shot_variant = shot_variant
            logging.info(f"Folder ID: {folderID}")
            self.load_csv_data()
            self.reduced_ontology = self.reduce_ontology(self.csv_file_path, self.ontology_file_path, self.folderID)
            self.prompt_instructions = self.load_prompt_from_file()

            if real_run == False :
                if mapping_language == "rml":
                    mapping_result = """This is your mapping: ```RML
                                @prefix rr: <http://www.w3.org/ns/r2rml#> .
                                @prefix rml: <http://semweb.mmlab.be/ns/rml#> .
                                @prefix ont: <http://example.com/PlantOntology#> .
                                @prefix ql: <http://semweb.mmlab.be/ns/ql#> .

                                <#PlantMapping>
                                    a rr:TriplesMap ;
                                    rml:logicalSource [
                                        rml:source "examples/data_sources/csv/data.csv" ;
                                        rml:referenceFormulation ql:CSV
                                    ] ;
                                    rr:subjectMap [
                                        rr:template "http://example.com/PlantOntology/{PlantID}" ;
                                        rr:class ont:Plant
                                    ] ;
                                    rr:predicateObjectMap [
                                        rr:predicate ont:plantId ;
                                        rr:objectMap [ rml:reference "PlantID" ]
                                    ] ;
                                    rr:predicateObjectMap [
                                        rr:predicate ont:plantName ;
                                        rr:objectMap [ rml:reference "PlantName" ]
                                    ] ;
                                    rr:predicateObjectMap [
                                        rr:predicate ont:hasLine ;
                                        rr:objectMap [
                                            rr:parentTriplesMap <#LineMapping>
                                        ]
                                    ] .

                                <#LineMapping>
                                    a rr:TriplesMap ;
                                    rml:logicalSource [
                                        rml:source "examples/data_sources/csv/data.csv" ;
                                        rml:referenceFormulation ql:CSV
                                    ] ;
                                    rr:subjectMap [
                                        rr:template "http://example.com/PlantOntology/{LineID}" ;
                                        rr:class ont:Line
                                    ] ;
                                    rr:predicateObjectMap [
                                        rr:predicate ont:lineId ;
                                        rr:objectMap [ rml:reference "LineID" ]
                                    ]  ;
                                    rr:predicateObjectMap [
                                        rr:predicate ont:lineName ;
                                        rr:objectMap [ rml:reference "LineName" ]
                                    ] .``` This is the end."""

                elif mapping_language == "yarrrml":
                    mapping_result = """
prefixes:
  ont: http://example.com/PlantOntology#

mappings:
  plant:
    sources:
      - [examples/data_sources/csv/generic/data/plant_line_machine.csv]
    s: ont:plant-$(PlantID)
    po:
      - [a, ont:Plant]
      - [ont:plantID, $(PlantID)]
      - [ont:plantName, $(PlantName)]
      - p: ont:hasLine
        o: 
          - mapping: line 
          
  line:
    sources:
      - [examples/data_sources/csv/generic/data/plant_line_machine.csv]
    s: ont:line-$(LineID)
    po:
      - [a, ont:Line]
      - [ont:lineId, $(LineID)]
      - [ont:lineName, $(LineName)] 
"""

            elif real_run == True and self.shot_variant == "few" :
                if not self.llm:
                    self.login_to_azure_llm()

                json_file_path = f"mapping-generation/input/{datasetname}/prompt_{mapping_language}/few_shot.json"
                
                few_shot_examples = self.get_few_shot_examples(json_file_path)
                if not few_shot_examples:
                    print("No valid few-shot examples found. Exiting.")
                    return

                logging.info(f"These are the few shot examples: {few_shot_examples}")

                input_ontology = self.load_txt_from_file(f"mapping-generation/generated_files/{datasetname}/{folderID}/reduced_ontology.ttl")
                input_data_list = self.load_csv_data()

                if datasize == "columns-only" :
                    input_data = ', '.join(map(str, input_data_list))
                elif datasize == "first20": 
                    input_data = input_data_list.to_string(index=False, header=True)    
                elif datasize == "full": 
                    input_data = input_data_list.to_string(index=False, header=True)   

                few_shot_prompt = self.get_few_shot_prompt(
                    few_shot_examples = few_shot_examples,
                    input_ontology = input_ontology,
                    input_data = input_data
                )

                output_parser = StrOutputParser()
                logging.info("Starting LLM chain for mapping generation with the best few shot example.")
                chain = self.construct_prompt(few_shot_prompt) | self.llm | output_parser

                debug_info = {
                    "input_data": input_data,
                    "input_ontology": input_ontology,
                }

                mapping_result = chain.invoke(debug_info)

                mapping_result = self.extract_mapping(mapping_result, mapping_language)

            elif real_run == True and self.shot_variant == "template":
                prompt_full = self.build_prompt_template_mapping()
                logging.info(f"Here is the zero shot prompt with mapping template: {prompt_full}.")
                mapping_result = self.generate_mapping(prompt_full=prompt_full)

            # NOT IMPLEMENTED FOR RML
            elif real_run == True and self.shot_variant == "zero" and mapping_language == "yarrrml":
                prompt_full = self.build_prompt_template_zero()
                logging.info(f"Here is the zero shot prompt: {prompt_full}.")
                mapping_result = self.generate_mapping(prompt_full=prompt_full)

            mapping_result = self.extract_mapping(mapping_result, mapping_language)
            logging.info(f"Generated Mappings: {mapping_result}")

            # NOT implemented for RML
            if mapping_language == "yarrrml" :
                mapping_result = self.add_unknown_prefixes(mapping_result, mapping_language)
                logging.info(f"Generated Mappings AFTER prefix/namespace check: {mapping_result}")

            return mapping_result
            
        except Exception as e:
            logging.error(f"Error in execution: {e}")
            raise

    def save_mapping_file(self, mappings:str, mapping_language, prefixes=None):
        if mapping_language == "rml":
            RML = URIRef("http://semweb.mmlab.be/ns/rml#")
            RR = URIRef("http://www.w3.org/ns/r2rml#")
            QL = URIRef("http://semweb.mmlab.be/ns/ql#")

            g = Graph()
            g.bind("rr", RR)
            g.bind("rml", RML)
            g.bind("ql", QL)
            g.bind("rdfs", RDFS)

            g.parse(data=mappings, format="turtle")
            
            script_path = Path(__file__).resolve()
            script_directory = script_path.parent

            file_name = f"generated_files/{self.datasetname}/{self.folderID}/generated_rml.ttl"

            mapping_path = script_directory.joinpath(file_name)

            try:
                if prefixes:
                    for prefix, namespace in prefixes.items():
                        g.bind(prefix, Namespace(namespace))
                        logging.debug(f"Binding prefix: {prefix} to namespace: {namespace}")

                g.serialize(destination=mapping_path, format='turtle') 

                logging.info(f"RML Mapping has been successfully serialized to Turtle format and saved at: {mapping_path}")

            except Exception as e:
                logging.error(f"Error serializing graph to Turtle: {e}")

        elif mapping_language == "yarrrml":
            yaml = YAML()
            yaml.indent(mapping=4, sequence=6, offset=3) 
            yaml.default_flow_style = False 

            data = yaml.load(mappings)
            
            script_path = Path(__file__).resolve()
            script_directory = script_path.parent

            file_name = f"generated_files/{self.datasetname}/{self.folderID}/generated_yarrrml.yml"
            mapping_path = script_directory.joinpath(file_name)

            with open(mapping_path, 'w') as file:
                    yaml.dump(data, file)

            logging.info(f"YARRRML Mapping has been successfully saved.")

    def evaluate_mapping(self, 
                         mapping_language: str):
        
        if mapping_language == "rml":
            logging.info(f"mapping-generation/generated_files/{datasetname}/{folderID}/generated_rml.ttl")
            evaluator = RMLEvaluator(f"mapping-generation/input/{datasetname}/prompt_rml/ontology.ttl", 
                                    f"mapping-generation/generated_files/{datasetname}/{folderID}/generated_rml.ttl", 
                                    f"mapping-generation/input/{datasetname}/reference_rml.ttl")

            precision, recall, f1, tp, fn, fp = evaluator.evaluate()

            explanation = evaluator.explain_evaluation_metrics(tp, fp, fn)
            print(explanation)
            
            experiment_docu = f"Execution;Data;Datasize;Ont reduction;shots;f1;precision;recall;TP;FN;FP\nRML Mapping Generator;{datasetname};{datasize};{ont_reduce_variant};[{shot_variant}];{f1};{precision};{recall};{tp};{fn};{fp}"

            experiment_docu_filepath = f"mapping-generation/generated_files/{datasetname}/{folderID}/experiment_documentation.csv"

            with open(experiment_docu_filepath, 'w') as file:
                file.write(experiment_docu)            

        elif mapping_language == "yarrrml":
            logging.info(f"mapping-generation/generated_files/{datasetname}/{folderID}/generated_yarrrml.yml")
            evaluator = YarrrmlEvaluator(f"mapping-generation/input/{datasetname}/prompt_yarrrml/ontology.ttl", 
                                    f"mapping-generation/generated_files/{datasetname}/{folderID}/generated_yarrrml.yml", 
                                    f"mapping-generation/input/{datasetname}/reference_yarrrml.yml")

            precision, recall, f1, tp, fn, fp = evaluator.evaluate()
            
            experiment_docu = f"Execution;Data;Datasize;Ont reduction;shots;f1;precision;recall;TP;FN;FP\nYARRRML Mapping Generator;{datasetname};{datasize};{ont_reduce_variant};[{shot_variant}];{f1};{precision};{recall};{tp};{fn};{fp}"

            experiment_docu_filepath = f"mapping-generation/generated_files/{datasetname}/{folderID}/experiment_documentation.csv"

            with open(experiment_docu_filepath, 'w') as file:
                file.write(experiment_docu)


# Execute everything - main usage #############################################################################
if __name__ == "__main__":

    datasetname = "generic" # Specify which dataset to run the generation on: "generic"
    mapping_language = "yarrrml" # Specify the language of the generated mapping: "yarrrml" or "rml"
    ont_reduce_variant = "naive" # Specify which ontology reduction method to employ: "llm" or "similar" or "naive"
    datasize = "first20" # Specify whether full data source or only column names are used from sample data source: "full", "columns-only", "first20"
    shot_variant = "few" # Define Mapping Template or Example Shots: enhancement of the prompt with template/shots: "zero", "template", "few"
    real_run = False # Configure whether to perform a real run executing an LLM or to work with dummy LLM responses

    generator = LLMMappingGenerator(datasetname, 
                                    mapping_language,
                                    ont_reduce_variant, 
                                    datasize, 
                                    shot_variant)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folderID = f"mapping-generation/generated_files/{datasetname}/{current_time}"
            
    try:
        os.mkdir(folderID)
        print(f"Directory '{folderID}' created successfully.")
    except FileExistsError:
        print(f"Directory '{folderID}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{folderID}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    folderID = f"{current_time}" # shorten the path to only the ID of the folder

    result = generator.run(folderID, real_run)
    
    generator.save_mapping_file(result, mapping_language, prefixes=None)
  
    generator.evaluate_mapping(mapping_language)