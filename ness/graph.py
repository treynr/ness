#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: parse.py
## desc: Graph building functions.
## auth: TR

from itertools import count
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from typing import Dict
from typing import List
from typing import Tuple
import logging
import networkx as nx
import numpy as np
import pandas as pd

from . import log
from .types import Gene
from .types import GeneSet
from .types import Homolog
from .types import Term
from .types import BioEntity
from .types import Inputs

logging.getLogger(__name__).addHandler(logging.NullHandler())


def encapsulate_bioentities2(
    annotations: pd.DataFrame = None,
    edges: pd.DataFrame = None,
    genesets: pd.DataFrame = None,
    ontologies: pd.DataFrame = None,
    homology: pd.DataFrame = None,
) -> tuple:
    """
    Wraps the parsed genes, sets, terms, and homologs in the BioEntity dataclass.

    arguments
        annotations: ontology or geneset annotations
        edges:       edges from bio networks
        genesets:    gene sets
        ontologies:  ontology relationships
        homology:    homologs

    returns
        a tuple containing wrapped bio entities
    """

    if annotations is not None:
        annotations['term'] = annotations.term.astype(str).map(Term)
        annotations['gene'] = annotations.gene.astype(str).map(Gene)

    ## Auto assume both nodes in the edge are genes. Should be true in the vast
    ## majority of cases.
    if edges is not None:
        edges['source'] = edges.astype(str).source.map(Gene)
        edges['sink'] = edges.astype(str).sink.map(Gene)

    if genesets is not None:
        genesets['gsid'] = genesets.astype(str).gsid.map(GeneSet)
        genesets['gene'] = genesets.astype(str).genes.map(Gene)

    if ontologies is not None:
        ontologies['child'] = ontologies.astype(str).child.map(Term)
        ontologies['parent'] = ontologies.astype(str).parent.map(Term)

    if homology is not None:
        homology['cluster'] = homology.astype(str).cluster.map(Homolog)
        homology['gene'] = homology.astype(str).gene.map(Gene)

    return (annotations, edges, genesets, ontologies, homology)


def encapsulate_bioentities(inputs: Inputs) -> Inputs:
    """
    Wraps the parsed genes, sets, terms, and homologs in the BioEntity dataclass.

    arguments
        inputs: Inputs dataclass containing various NESS user inputs

    returns
        a modified Inputs dataclass containing user inputs encapsulated in the
        BioEntity dataclass
    """

    if inputs.annotations is not None:
        inputs.annotations['term'] = (
            inputs.annotations[['term', 'biotype1']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )
        inputs.annotations['gene'] = (
            inputs.annotations[['gene', 'biotype2']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )

    if inputs.edges is not None:
        inputs.edges['source'] = (
            inputs.edges[['source', 'biotype1']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )
        inputs.edges['sink'] = (
            inputs.edges[['sink', 'biotype2']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )

    if inputs.genesets is not None:
        inputs.genesets['geneset'] = (
            inputs.genesets[['geneset', 'biotype1']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )
        inputs.genesets['gene'] = (
            inputs.genesets[['gene', 'biotype2']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )

    if inputs.ontologies is not None:
        inputs.ontologies['child'] = (
            inputs.ontologies[['child', 'biotype1']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )
        inputs.ontologies['parent'] = (
            inputs.ontologies[['parent', 'biotype2']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )

    if inputs.homology is not None:
        inputs.homology['cluster'] = (
            inputs.homology[['cluster', 'biotype1']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )
        inputs.homology['gene'] = (
            inputs.homology[['gene', 'biotype2']]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
        )

    return inputs


def generate_node_uids(inputs: Inputs) -> Dict[BioEntity, int]:
    """
    Generate UIDs for bioentities from the given relationships.

    arguments
        inputs: an Inputs dataclass containing various NESS user inputs

    returns
        a dict mapping bioentities to their UIDs
    """

    entities: List[BioEntity] = []

    if inputs.annotations is not None:
        entities.extend(inputs.annotations.term.tolist())
        entities.extend(inputs.annotations.gene.tolist())

    if inputs.edges is not None:
        entities.extend(inputs.edges.source.tolist())
        entities.extend(inputs.edges.sink.tolist())

    if inputs.genesets is not None:
        entities.extend(inputs.genesets.geneset.tolist())
        entities.extend(inputs.genesets.gene.tolist())

    if inputs.ontologies is not None:
        entities.extend(inputs.ontologies.child.tolist())
        entities.extend(inputs.ontologies.parent.tolist())

    if inputs.homology is not None:
        entities.extend(inputs.homology.cluster.tolist())
        entities.extend(inputs.homology.gene.tolist())

    entities = pd.Series(entities).drop_duplicates()

    return dict(zip(entities, count()))


def shuffle_node_labels(uids: Dict[BioEntity, int]) -> Dict[BioEntity, int]:
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
    uids: Dict[BioEntity, int],
    inputs: Inputs,
    undirected: bool = True
) -> nx.DiGraph:
    """
    Build the heterogeneous graph.

    arguments
        uids:       node UID map
        inputs:     an Inputs dataclass containing various NESS user inputs
        undirected: if true use undirected edges

    returns
        the hetnet
    """

    ## the python "type" system is fucking dumb
    relations = []  # type: ignore

    if inputs.annotations is not None:
        relations.extend(inputs
            .annotations[['term', 'gene', 'score']]
            .itertuples(index=False, name=None)
        )

    if inputs.edges is not None:
        relations.extend(inputs
            .edges[['source', 'sink', 'score']]
            .itertuples(index=False, name=None)
        )

    if inputs.genesets is not None:
        relations.extend(inputs
            .genesets[['geneset', 'gene', 'score']]
            .itertuples(index=False, name=None)
        )

    if inputs.ontologies is not None:
        relations.extend(inputs
            .ontologies[['child', 'parent', 'score']]
            .itertuples(index=False, name=None)
        )

    if inputs.homology is not None:
        relations.extend(inputs
            .homology[['cluster', 'gene', 'score']]
            .itertuples(index=False, name=None)
        )

    mapped_relations = [(uids[a], uids[b], s) for a, b, s in relations]

    if undirected:
        mapped_relations.extend([(uids[b], uids[a], s) for a, b, s in relations])

    graph = nx.DiGraph()

    for e1, e2, w in mapped_relations:
        graph.add_edge(e1, e2, weight=w)
    # graph.add_edges_from(mapped_relations)

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

    # for ef, et in graph.edges:
    #     matrix[ef, et] = 1.0

    for ef, et, d in graph.edges.data():
        matrix[ef, et] = d['weight']

    return matrix


def old_column_normalize_matrix(matrix: csr_matrix) -> csr_matrix:
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


def column_normalize_matrix(matrix: csr_matrix) -> csr_matrix:
    """
    Column normalize the transition probability matrix.

    arguments
        matrix: graph matrix

    returns
        normalized matrix
    """

    ## We do this ugly fucking thing cause it's very, very fast. Like seriously, it's
    ## a 25x improvement in speed compared to looping
    matrix.data = (
        matrix.data /
        np.repeat(
            np.add.reduceat(matrix.data, matrix.indptr[:-1]),
            np.diff(matrix.indptr)
        )
    )

    return matrix


def build_graph(inputs: Inputs) -> Tuple:
    """
    Build the heterogeneous graph from NESS inputs.

    argumens
        inputs: parsed NESS inputs

    returns
        a tuple containing the node UID map and the heterogeneous graph
    """

    log._logger.info('Generating node UIDs...')

    wrapped_inputs = encapsulate_bioentities(inputs)
    uids = generate_node_uids(wrapped_inputs)

    log._logger.info('Building the heterogeneous graph...')

    hetnet = build_heterogeneous_graph(uids, wrapped_inputs)

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
    matrix = csr_matrix(matrix)
    matrix = column_normalize_matrix(matrix)

    return matrix


def save_graph(hetnet: nx.Graph, uids: Dict[BioEntity, int], output: Path) -> None:
    """
    Save the heterogeneous graph to a file.

    arguments
        hetnet: the graph
        uids:   UID mapping
        output: output path
    """

    ## Reverse mapping
    ruids = dict([(b, a) for a, b in uids.items()])
    edges = []

    for e1, e2 in hetnet.edges.keys():
        e1 = ruids[e1]
        e2 = ruids[e2]

        edges.append([e1.id, e2.id, e1.biotype, e2.biotype])

    pd.DataFrame(edges, columns=['node1', 'node2', 'biotype1', 'biotype2']).to_csv(
        output, sep='\t', index=False
    )

