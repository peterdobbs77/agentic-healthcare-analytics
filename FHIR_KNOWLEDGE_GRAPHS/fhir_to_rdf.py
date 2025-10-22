# Author: Peter N. Dobbs
# Influenced by https://github.com/BD2KOnFHIR/fhirtordf, but improved for my purposes

from rdflib import Graph, URIRef, BNode, OWL, RDF, Literal, Namespace
import networkx as nx
import matplotlib.pyplot as plt
import warnings

# NAMESPACES for graph
FHIR = Namespace("http://hl7.org/fhir/")
EX = Namespace("http://example.org/")

SKIP_KEYS = ['resourceType', 'id',  # because these two should already have been parsed
             # TODO: setup handling for the rest of these properties
             'extension', 'communication', 'telecom', 'address', 'maritalStatus', 'language',
             'participant', 'individual', 'verificationStatus',
             'clinicalStatus', 'period', 'class' 
            ]

# map json key value to resource type
MAP_KEY_TO_RESOURCE_TYPE = { # assumes that these keys are a reference to only one resource type (not necessarily true)
    'subject': 'Patient',
    'patient': 'Patient',
    'serviceProvider': 'Organization',
    'encounter': 'Encounter'
}

def pretty_view_rdf_graph(self, title="Simplified View of RDF Representation of FHIR Resources"):
    '''Visualize simplified view of the RDF graph,
        just showing connections between FHIR Resources
        Does not include all the properties connections
    :param g: `rdflib.Graph`'''
    # Create NetworkX graph
    nx_graph = nx.DiGraph()

    for subj, pred, obj in self._graph:
        if not isinstance(obj, URIRef):
            continue
        # Convert nodes to readable strings
        subj_str = str(subj).split('/')[-2] + '\n' + str(subj).split('/')[-1]
        pred_str = str(pred).split('/')[-1]
        obj_str = str(obj).split('/')[-2] + '\n' + str(obj).split('/')[-1]

        # Add edge with predicate label
        nx_graph.add_edge(subj_str, obj_str, label=pred_str)
    
    # Draw graph
    # pos = nx.arf_layout(nx_graph, seed=42) # ARF is better than spring
    pos = nx.shell_layout(nx_graph, 
                            nlist=[[x for x in nx_graph.nodes() if 'fhir' in x],
                                    [x for x in nx_graph.nodes() if 'fhir' not in x]])
    edge_labels = nx.get_edge_attributes(nx_graph, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title)
    plt.tight_layout()
    plt.show()

class FhirGraph():
    '''A FHIR RDF representation of a FHIR JSON resource'''

    def __init__(self, data, target_graph: Graph=None) -> Graph:
        '''Construct an RDF representaiton of FHIR Resource(s)
        :param data: source JSON representation of FHIR Resource(s)
        '''
        self._graph = target_graph if target_graph is not None else Graph()
        self._graph.bind("fhir", FHIR)
        self._graph.bind("rdf", RDF)
        self._graph.bind("ex", EX)
        self._source = data
        if 'entry' in self._source:
            # Bundle
            self._root_resource_type = 'Bundle'
        elif 'resourceType' in self._source:
            self._root_resource_type = self._source['resourceType']

    def add_direct_reference_to_resource(self, root_uri: URIRef, data_key: str, data_val: dict, referenced_resource_type: str):
        '''
        :param root_uri: root level of resource in RDF
        :param data_key: key for JSON object reference, should be 'subject'
        :param data_val: value for JSON object reference as dictionary
        :param referenced_resource_type: must be from the FHIR Resource types
        '''
        # OPTION 1: establish direct link
        if 'reference' in data_val:
            reference_id = str.split(data_val['reference'], ':')[-1] # assuming reference is of the form "urn:uuid:#######"
            reference_uri = URIRef(EX[referenced_resource_type + '/' + reference_id])
            self._graph.add((root_uri, FHIR[data_key], reference_uri))

    def _add_blank_reference_node_to_resource(self, root_uri: URIRef, data_key: str, data_val: dict, referenced_resource_type: str):
        '''Establishes linkage via a blank node with properties of the reference
        :param root_uri: root level of resource in RDF
        :param data_key: key for JSON object reference, should be 'subject'
        :param data_val: value for JSON object reference as dictionary
        :param referenced_resource_type: must be from the FHIR Resource types
        '''
        warnings.warn("Warning: This method is unstable. Use `add_direct_reference_to_resource` instead", DeprecationWarning)
        # OPTION 2: establish link via blank node
        resource_reference = BNode()
        self._graph.add((root_uri, FHIR[data_key], resource_reference))
        self._graph.add((resource_reference, RDF.type, FHIR[referenced_resource_type]))
        self._graph.add((resource_reference, FHIR.type, FHIR[referenced_resource_type]))
        if 'reference' in data_val:
            # establish blank node reference
            reference_node = BNode()
            self._graph.add((resource_reference, FHIR.reference, reference_node))
            self._graph.add((reference_node, FHIR.value, Literal(data_val['reference'])))
            # establish fhir:link
            # TODO: handle other reference schema formats (e.g., "Resource/identifier")
            reference_id = str.split(data_val['reference'], ':')[-1] # assuming reference is of the form "urn:uuid:#######"
            reference_uri = URIRef(EX[referenced_resource_type + '/' + reference_id])
            self._graph.add((resource_reference, FHIR.link, reference_uri))
        if 'display' in data_val:
            self._graph.add((resource_reference, FHIR.display, Literal(data_val['display'])))
        print([x for x in data_val.items()]) # TEMP DEBUG LINE

    def add_data_to_graph(self, data, root_uri=None):
        '''Add data from FHIR resource to graph
        :param data: data from a FHIR resource to be added to the graph
        :param root_uri: root level of resource in RDF, for recursive calls
        '''
        # print(data)   # TEMP DEBUG LINE
        if root_uri is None:
            resource_type = data['resourceType']
            root_uri = URIRef(EX[resource_type + '/' + data['id']])
            
            # Add basic properties for each resource
            self._graph.add((root_uri, RDF.type, FHIR[resource_type]))
            self._graph.add((root_uri, FHIR.nodeRole, FHIR.treeRoot))
            self._graph.add((root_uri, FHIR.resourceType, Literal(resource_type)))
        
        # Iterate over the attributes of the resource
        for key, value in data.items():
            if key in SKIP_KEYS:  # Skip resourceType, id, etc
                continue

            if isinstance(value, dict):  # Handle nested objects
                if key in MAP_KEY_TO_RESOURCE_TYPE.keys():
                    self.add_direct_reference_to_resource(root_uri, key, value, MAP_KEY_TO_RESOURCE_TYPE[key])
                else:
                    subNode = BNode()
                    self._graph.add((root_uri, FHIR[key], subNode))  # nest a node for the key-value pair
                    self.add_data_to_graph(value, subNode)  # Recursive call for nested objects
            elif isinstance(value, list):  # Handle lists
                for item in value:
                    if isinstance(item, dict):  # If the item is a resource
                        subNode = BNode()
                        self._graph.add((root_uri, FHIR[key], subNode))
                        self.add_data_to_graph(item, subNode)  # Recursive call for nested resources
                    else:
                        # key in ["given", "prefix", ...]
                        subNode = BNode()
                        self._graph.add((root_uri, FHIR[key], subNode))
                        self._graph.add((subNode, FHIR.value, Literal(item)))
            else:
                if value == "{score}":
                    continue
                # handle primitives
                elif key in ["status", "code", "unit", "system", "value", "active", "gender"]:
                    subNode = BNode()
                    self._graph.add((root_uri, FHIR[key], subNode))
                    self._graph.add((subNode, FHIR.value, Literal(value)))
                else:
                    self._graph.add((root_uri, FHIR[key], Literal(value)))

    def generate(self, limit=5) -> Graph:
        '''generator method for FhirGraph'''
        i = 0
        for entry in self._source['entry']:
            resource = entry['resource']
            self.add_data_to_graph(resource, None)
            i += 1
            if i > limit:
                break
        return self._graph
        # self.add_data_to_graph(data=self._source)

import unittest
import json
import glob

class Test_FhirGraph(unittest.TestCase):
    """Test cases for generating RDF graph from FHIR JSON Bundle"""

    def test_parse_fhirbundle_to_rdfgraph(self):
        """
        Simple test to ensure FhirGraph is creating valid RDF
        """
        sample_bundle_path="./fhir/Alethea978_Crooks415_78480da7-7361-4c99-bfb9-339a403d9ae1.json"

        large_multi_graph = Graph()
        
        with open(sample_bundle_path, mode='r', encoding='utf-8') as f:
            fhir_bundle = json.load(f)
            large_multi_fhirgraph = FhirGraph(fhir_bundle, large_multi_graph)
            large_multi_graph = large_multi_fhirgraph.generate()
        
        print(len(large_multi_graph))

        query="""
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o
        }
        LIMIT 5
        """
        self.assertIsNotNone(large_multi_graph.query(query))


    def test_parse_many_fhirbundle_to_rdfgraph(self):
        """
        Scaling up simple test to many fhir bundles
        Note: this individual test takes over 20 seconds to run
        """
        
        synthea_bundles = glob.glob("./fhir/*.json")

        large_multi_graph = Graph()

        for fpath in synthea_bundles:
            with open(fpath, mode='r', encoding='utf-8') as f:
                fhir_bundle = json.load(f)
                large_multi_fhirgraph = FhirGraph(fhir_bundle, large_multi_graph)
                large_multi_graph = large_multi_fhirgraph.generate()
        
        print(len(large_multi_graph))

        query="""
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o
        }
        LIMIT 5
        """
        self.assertIsNotNone(large_multi_graph.query(query))


if __name__ == '__main__':
    unittest.main()