# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import yaml
from rdflib import Graph, URIRef, RDF, OWL
from sklearn.metrics import precision_score, recall_score, f1_score
from urllib.parse import urlencode
import os

class YarrrmlEvaluator:
    def __init__(self, ontology_file, generated_yarrrml_file, reference_yarrrml_file):
        self.ontology_graph = Graph()
        self.generated_yarrrml = generated_yarrrml_file
        self.reference_yarrrml = reference_yarrrml_file

        self.ontology_graph.parse(ontology_file, format="turtle") 

        if not os.path.exists(generated_yarrrml_file):
            raise FileNotFoundError(f"The generated YARRRML file does not exist: {generated_yarrrml_file}")
        if not os.path.exists(reference_yarrrml_file):
            raise FileNotFoundError(f"The reference YARRRML file does not exist: {reference_yarrrml_file}")

        self.generated_yarrrml = self.load_yarrrml(generated_yarrrml_file)
        self.reference_yarrrml = self.load_yarrrml(reference_yarrrml_file)

        self.ontology_uris = self.extract_ontology_uris()

    def load_yarrrml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def extract_ontology_uris(self):
        classes = set()
        object_properties = set()
        datatype_properties = set()

        for subj, pred, obj in self.ontology_graph:
            if pred == RDF.type and obj == OWL.Class:
                classes.add(str(subj))
            elif pred == RDF.type and obj == OWL.ObjectProperty:
                object_properties.add(str(subj))
            elif pred == RDF.type and obj == OWL.DatatypeProperty:
                datatype_properties.add(str(subj))
            elif pred == RDF.type and obj == RDF.Property:
                object_properties.add(str(subj)) 

        return classes.union(object_properties).union(datatype_properties)

    def normalize_uri(self, uri, prefix_map):
        if ':' in uri and not uri.startswith('http'):
            prefix, local_part = uri.split(':', 1)
            if prefix in prefix_map:
                base_uri = prefix_map[prefix]

                if not base_uri.endswith('/') and not base_uri.endswith('#'):
                    base_uri += '#'  # Append '#' if it doesn't already end with '/' or '#'
                
                full_uri = base_uri + local_part
                return full_uri 
            else:
                print(f"Prefix '{prefix}' not found in prefix map.")
        return uri

    def extract_triple_patterns(self, yarrrml_data: dict, prefix_map: dict) -> set:
        triple_patterns = set()
        mappings = yarrrml_data.get('mappings', {})

        for mapping_name, mapping_content in mappings.items():
            predicate_object_list = mapping_content.get('po', [])

            for po in predicate_object_list:
                try:
                    if isinstance(po, dict):
                        predicate = po.get('predicate', None)
                        obj = po.get('object', None)
                    elif isinstance(po, list) and len(po) >= 2:
                        predicate, obj = po[0], po[1]
                    else:
                        print(f"Skipping unexpected 'po' structure in mapping '{mapping_name}': {po}")
                        continue

                    if predicate == 'a':
                        obj_uri = self.normalize_uri(obj, prefix_map)
                        triple_patterns.add((None, 'a', obj_uri))
                    else:
                        pred_uri = self.normalize_uri(predicate, prefix_map)
                        obj_uri = self.normalize_uri(obj, prefix_map)
                        triple_patterns.add((None, pred_uri, obj_uri))

                except Exception as e:
                    print(f"Error processing 'po' in mapping '{mapping_name}': {e}")

        return triple_patterns


    def validate_generated_triples(self, triples):
        valid_triples = set()

        for _, predicate, obj in triples:
            if predicate == 'a':
                if obj in self.ontology_uris:
                    valid_triples.add((None, predicate, obj))
                else:
                    print(f"Class '{obj}' not found in ontology.")

            else:
                if predicate in self.ontology_uris:
                    valid_triples.add((None, predicate, obj))
                else:
                    print(f"Predicate '{predicate}' not found in ontology.")
        
        return valid_triples

    def evaluate(self) -> tuple[float, float, float, int, int, int]:
        prefix_map = self.generated_yarrrml.get('prefixes', {})

        generated_triples = self.extract_triple_patterns(self.generated_yarrrml, prefix_map)
        reference_triples = self.extract_triple_patterns(self.reference_yarrrml, prefix_map)
        valid_generated_triples = self.validate_generated_triples(generated_triples)

        tp = len(valid_generated_triples.intersection(reference_triples))
        fp = len(valid_generated_triples.difference(reference_triples))
        fn = len(reference_triples.difference(valid_generated_triples))

        y_true = [1] * len(reference_triples) + [0] * fp

        y_pred = [1 if triple in valid_generated_triples else 0 for triple in reference_triples] + [1] * fp

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return precision, recall, f1, tp, fn, fp


    def explain_evaluation_metrics(self, tp: int, fp: int, fn: int) -> str:
        explanation = []

        explanation.append(f"True Positives (TP): {tp}")
        explanation.append("These are the triples that were correctly generated and also exist in the reference data.")

        explanation.append(f"False Positives (FP): {fp}")
        explanation.append(
            "These are the triples that were generated but do not exist in the reference data. "
        )

        explanation.append(f"False Negatives (FN): {fn}")
        explanation.append(
            "These are the triples that exist in the reference data but were not generated. "
        )

        explanation.append(
            "Precision is the ratio of correctly generated triples (TP) out of all generated triples (TP + FP). "
        )
        explanation.append(
            "Recall is the ratio of correctly generated triples (TP) out of all reference triples (TP + FN). "
        )
        explanation.append(
            "F1 Score is the harmonic mean of precision and recall, balancing both metrics."
        )

        return "\n".join(explanation)

