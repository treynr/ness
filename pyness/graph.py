#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: parse.py
## desc: Graph building functions.
## auth: TR

from itertools import count
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from typing import Dict
import logging
import networkx as nx
import numpy as np
import pandas as pd

from . import log
from .types import Gene
from .types import GeneSet
from .types import Homolog
from .types import Term

logging.getLogger(__name__).addHandler(logging.NullHandler())


def generate_node_uids(
    annotations: pd.DataFrame = None,
    edges: pd.DataFrame = None,
    genesets: pd.DataFrame = None,
    ontologies: pd.DataFrame = None,
    homology: pd.DataFrame = None,
) -> Dict[str, int]:
    """
    Generate UIDs for bio-entities from the given relationships.

    arguments
        annotations: ontology or geneset annotations
        edges:       edges from bio networks
        genesets:    gene sets
        ontologies:  ontology relationships
        homology:    homologs

    returns
        a dict mapping bio-entities to their UIDs
    """

    entities = []

    if annotations is not None:
        entities.extend([Term(t) for t in annotations.term.tolist()])
        entities.extend([Gene(g) for g in annotations.gene.tolist()])

    if edges is not None:
        entities.extend([Gene(g) for g in edges.node_from.tolist()])
        entities.extend([Gene(g) for g in edges.node_to.tolist()])

    if genesets is not None:
        entities.extend([GeneSet(g) for g in genesets.gsid.tolist()])
        entities.extend([Gene(g) for g in genesets.gene.tolist()])

    if ontologies is not None:
        entities.extend([Term(t) for t in ontologies.child.tolist()])
        entities.extend([Term(t) for t in ontologies.parent.tolist()])

    if homology is not None:
        entities.extend([Homolog(h) for h in homology.cluster.tolist()])
        entities.extend([Gene(g) for g in homology.gene.tolist()])

    entities = pd.Series(entities).drop_duplicates()

    return dict(zip(entities, count()))


def shuffle_node_labels(uids: Dict[str, int]) -> Dict[str, int]:
    """
    Shuffle graph node labels for permutation testing.

    arguments
        uids: the UID map

    returns
        a UID map
    """

    labels = np.random.permutation(list(uids.keys()))

    return dict(zip(labels, list(uids.values())))


def build_heterogeneous_graph(
    uids: Dict[str, int],
    annotations: pd.DataFrame = None,
    edges: pd.DataFrame = None,
    genesets: pd.DataFrame = None,
    ontologies: pd.DataFrame = None,
    homology: pd.DataFrame = None,
    undirected: bool = True
) -> nx.DiGraph:
    """
    Build the heterogeneous graph.

    arguments
        uids:        node UID map
        annotations: ontology or geneset annotations
        edges:       edges from bio networks
        genesets:    gene sets
        ontologies:  ontology relationships
        homology:    homologs

    returns
        the hetnetwork
    """

    relations = []

    if annotations is not None:
        relations.extend(
            annotations[['term', 'gene']].itertuples(index=False, name=None)
        )

    if edges is not None:
        relations.extend(
            edges[['node_from', 'node_to']].itertuples(index=False, name=None)
        )

    if genesets is not None:
        relations.extend(
            genesets[['gsid', 'gene']].itertuples(index=False, name=None)
        )

    if ontologies is not None:
        relations.extend(
            ontologies[['child', 'parent']].itertuples(index=False, name=None)
        )

    if homology is not None:
        relations.extend(
            ontologies[['cluster', 'gene']].itertuples(index=False, name=None)
        )

    mapped_relations = [(uids[a], uids[b]) for a, b in relations]

    if undirected:
        mapped_relations.extend([(uids[b], uids[a]) for a, b in relations])

    graph = nx.DiGraph()

    graph.add_edges_from(mapped_relations)

    return graph


def graph_to_sparse_matrix(graph: nx.Graph) -> dok_matrix:
    """
    Convert a graph to a DOK sparse matrix.

    arguments
        graph: networkx graph

    returns
        a sparse matrix
    """

    nodes = nx.number_of_nodes(graph)
    matrix = dok_matrix((nodes, nodes), dtype=np.double)

    for ef, et in graph.edges:
        matrix[ef, et] = 1.0

    return matrix


def column_normalize_matrix(matrix: csr_matrix) -> csr_matrix:
    """
    Column normalize the transition probability matrix.

    arguments
        matrix: graph matrix

    returns
        normalized matrix
    """

    for i in range(matrix.shape[1]):
        if matrix[i, :].sum() != 0:
            matrix[i, :] = matrix[i, :] / matrix[i, :].sum()

    return matrix


def build_graph(inputs: Dict[str, pd.DataFrame]) -> nx.Graph:
    """
    Build the heterogeneous graph from NESS inputs.

    argumens
        inputs: parsed NESS inputs

    returns
        a tuple containing the node UID map and the heterogeneous graph
    """

    log._logger.info('Generating node UIDs...')

    uids = generate_node_uids(
        annotations=inputs.annotations,
        edges=inputs.edges,
        genesets=inputs.genesets,
        ontologies=inputs.ontologies,
        homology=inputs.homology
    )

    log._logger.info('Building the heterogeneous graph...')

    hetnet = build_heterogeneous_graph(
        uids,
        annotations=inputs.annotations,
        edges=inputs.edges,
        genesets=inputs.genesets,
        ontologies=inputs.ontologies,
        homology=inputs.homology
    )

    return (uids, hetnet)


def build_matrix(hetnet: nx.Graph) -> csr_matrix:
    """
    Build and normalize the transition probability matrix.

    arguments
        hetnet: the hetnet graph

    returns
        the prob. matrix
    """

    log._logger.info('Building the transition probability matrix...')

    matrix = graph_to_sparse_matrix(hetnet)

    log._logger.info('Normalizing the transition matrix...')

    ## Convert to a CSR matrix prior to normalization for faster slicing
    matrix = matrix.tocsr()
    matrix = column_normalize_matrix(matrix)

    return matrix

