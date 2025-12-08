# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import rdflib
from rdflib import Graph, URIRef, RDF, OWL, Literal
from rdflib.namespace import RDFS, Namespace
from sklearn.metrics import precision_score, recall_score, f1_score
import os

class RMLEvaluator:
    def __init__(self, ontology_file, generated_rml_file, reference_rml_file):
        self.ontology_graph = Graph()
        self.generated_rml_file = generated_rml_file
        self.reference_rml_file = reference_rml_file

        self.ontology_graph.parse(ontology_file, format="turtle") 

        if not os.path.exists(generated_rml_file):
            raise FileNotFoundError(f"The generated RML file does not exist: {generated_rml_file}")
        if not os.path.exists(reference_rml_file):
            raise FileNotFoundError(f"The reference RML file does not exist: {reference_rml_file}")

        self.generated_rml = self.load_rml(generated_rml_file)
        self.reference_rml = self.load_rml(reference_rml_file)

        self.ontology_uris = self.extract_ontology_uris()

    def load_rml(self, file_path):
        g = Graph()
        g.parse(file_path, format="turtle")
        return g

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

    def extract_triple_patterns(self, rml_graph: Graph) -> set:
        triple_patterns = set()
        RR = Namespace("http://www.w3.org/ns/r2rml#")
        RML = Namespace("http://semweb.mmlab.be/ns/rml#")
        
        for triples_map in rml_graph.subjects(RDF.type, RR.TriplesMap):
            for subject_map in rml_graph.objects(triples_map, RR.subjectMap):
                for _class in rml_graph.objects(subject_map, RR["class"]):
                    triple_patterns.add((None, str(RDF.type), str(_class)))

            for pom in rml_graph.objects(triples_map, RR.predicateObjectMap):
                for predicate in rml_graph.objects(pom, RR.predicate):
                    for obj in rml_graph.objects(pom, RR.objectMap):
                        for parent_triples_map in rml_graph.objects(obj, RR.parentTriplesMap):
                            triple_patterns.add((None, str(predicate), str(parent_triples_map)))
                        for constant in rml_graph.objects(obj, RR.constant):
                            triple_patterns.add((None, str(predicate), str(constant)))
                        for reference in rml_graph.objects(obj, RML.reference):
                            triple_patterns.add((None, str(predicate), str(reference))) 
                        for _template in rml_graph.objects(obj, RR.template):
                            triple_patterns.add((None, str(predicate), str(_template)))
        return triple_patterns

    def validate_generated_triples(self, triples):
        valid_triples = set()

        for _, predicate, obj in triples:
            if predicate == str(RDF.type):
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
        generated_triples = self.extract_triple_patterns(self.generated_rml)
        reference_triples = self.extract_triple_patterns(self.reference_rml)

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

