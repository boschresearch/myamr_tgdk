# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import pandas as pd
import rdflib
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD
import logging
from datetime import datetime
import os

class OntoReducerNaive:
    def __init__(self, 
                 csv_file, 
                 ontology_file, 
                 datasetname, 
                 folderID):

        self.csv_file = csv_file
        self.ontology_file = ontology_file
        self.datasetname = datasetname
        self.folderID = folderID
        self.namespace = Namespace("http://example.com/PlantOntology#")


    def load_ontology_file(self, ontology_file) -> rdflib.Graph:
        """Load ontology from a Turtle (.ttl) file and return an RDFLib graph."""
        self.ontology_file = ontology_file

        try:
            logging.info(f"Loading ontology from Turtle file.")
            g = rdflib.Graph()
            g.parse(self.ontology_file, format='turtle')
            return g
        except Exception as e:
            logging.error(f"Failed to load ontology file: {e}")
            raise

    def reduce_ontology(self, csv_file, ontology_graph: rdflib.Graph) -> Graph:
        """
        Restructure the ontology to focus on relevant information and reduce token consumption.
        """
        self.csv_file = csv_file
        
        logging.info("Reducing the ontology based on the subset query.")


        subset_query = """
            CONSTRUCT {
                ?s a ?class .
                ?s rdfs:label ?label .
                ?s rdf:type ?type .
                ?s rdfs:comment ?comment .
                }
            WHERE {
                ?s a ?class .
                OPTIONAL{
                    ?s rdfs:label ?label .
                }
                OPTIONAL{
                    ?s rdfs:comment ?comment .
                }
            }
        """

        qres = ontology_graph.query(subset_query)

        reduced_ontology_graph = rdflib.Graph()

        for prefix, namespace in ontology_graph.namespaces():
            reduced_ontology_graph.bind(prefix, namespace)

        for triple in qres:
            reduced_ontology_graph.add(triple)

        context_ontology = reduced_ontology_graph.serialize(format='turtle')

        context_ontology = context_ontology.replace('    ', '')

        return context_ontology



    def save_reduced_ontology(self, reduced_ontology, datasetname, folderID):

        self.datasetname = datasetname
        self.folderID = folderID
        self.reduced_ontology = reduced_ontology

        reduced_ont_file_path = f"mapping-generation/generated_files/{datasetname}/{folderID}/reduced_ontology.ttl"

        with open(reduced_ont_file_path, 'w') as file:
            file.write(self.reduced_ontology)


    def run(self, datasetname, folderID, csv_file, ontology_file):
        ontology = self.load_ontology_file(ontology_file)
        reduced_ontology = self.reduce_ontology(csv_file, ontology)
        self.save_reduced_ontology(reduced_ontology, datasetname, folderID)

        return reduced_ontology
    
