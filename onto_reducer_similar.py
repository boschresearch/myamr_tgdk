# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import logging
from datetime import datetime
import os
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import RDF, Graph, URIRef, Literal
import numpy as np


class OntoReducerSim():
    def __init__(self, 
                 csv_file: str, 
                 ontology_file: str, 
                 datasetname: str,
                 folderID: str
                 ):
        
        self.csv_file = csv_file
        self.ontology_file = ontology_file
        self.datasetname = datasetname
        self.folderID = folderID
        self.csv_data = pd.read_csv(csv_file)
        self.ontology_graph = Graph()
        self.ontology_graph.parse(ontology_file, format='turtle')
        self.reduced_graph = Graph()
        self.relevant_classes = set()
        self.relevant_properties = set()
        self.class_labels = set()
        self.property_labels = set()

    def load_data(self, csv_file):
        """Load the dataset from a CSV file."""
        return pd.read_csv(csv_file)

    def load_ontology(self, ontology_file):
        """Load the ontology from a TTL file and extract terms."""
        g = Graph()
        g.parse(ontology_file, format='turtle')
        
        ontology_terms = [str(s) for s in g.subjects()]
        return pd.DataFrame(ontology_terms, columns=['ontology_term'])

    def embed_texts(self, texts, embedder):
        embeddings = embedder(texts)
        
        return np.array([embedding[0][0] for embedding in embeddings])  

    def reduce_ontology(self, dataset, ontology, embedder, similarity_threshold=0.2) -> Graph:

        column_headers = dataset.columns.tolist()

        column_embeddings = self.embed_texts(column_headers, embedder)
        ontology_embeddings = self.embed_texts(ontology['ontology_term'].tolist(), embedder)

        similarities = cosine_similarity(column_embeddings, ontology_embeddings)

        relevant_ontology_terms = set()
        for i, column_header in enumerate(column_headers):
            for j, ontology_term in enumerate(ontology['ontology_term']):
                if similarities[i][j] >= similarity_threshold:
                    relevant_ontology_terms.add(ontology_term)

        reduced_graph = Graph()

        for prefix, namespace in self.ontology_graph.namespaces():
            reduced_graph.bind(prefix, namespace)

        for term in relevant_ontology_terms:
            reduced_graph.add((URIRef(term), RDF.type, Literal(term)))  

        for subj, pred, obj in self.ontology_graph:
            if str(subj) in relevant_ontology_terms or str(obj) in relevant_ontology_terms:
                reduced_graph.add((subj, pred, obj))

            ttl_output = reduced_graph.serialize(format='turtle')

        return ttl_output

    def save_reduced_ontology(self, red_ont, datasetname, folderID):
    
        reduced_graph = Graph()
        reduced_graph.parse(data=red_ont, format='turtle')

        output_reuse = f"mapping-generation/generated_files/{datasetname}/{folderID}/reduced_ontology.ttl"
        
        reduced_graph.serialize(destination=output_reuse, format='turtle')
        logging.info("Reduced ontology successfully saved.")



    def run(self):
        dataset = self.load_data(self.csv_file)
        logging.info("Data set successfully loaded.")
        ontology = self.load_ontology(self.ontology_file)
        logging.info("Ontology successfully loaded.")

        embedder = pipeline('feature-extraction', model='distilbert-base-uncased')

        red_ont = self.reduce_ontology(dataset, ontology, embedder)

        self.save_reduced_ontology(red_ont, self.datasetname, self.folderID)

        return red_ont

