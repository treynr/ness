#!/usr/bin/env python
# -*- coding: utf8 -*-

## file: test_graph.py
## desc: Test functions in graph.py.

from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ness import graph
from ness import parse
from ness import types

@pytest.fixture(scope='module')
def root_dir():
    return Path(__file__).resolve().parent.as_posix()

@pytest.fixture(scope='module')
def sample_edges(root_dir):
    return parse.read_edges(Path(root_dir, 'data/edge-sample0.tsv').as_posix())

@pytest.fixture(scope='module')
def sample_annotations(root_dir):
    return parse.read_annotations(
        Path(root_dir, 'data/annotation-sample0.tsv').as_posix()
    )

@pytest.fixture(scope='module')
def sample_homology(root_dir):
    return parse.read_homology(Path(root_dir, 'data/homology-sample0.tsv').as_posix())

@pytest.fixture(scope='module')
def sample_geneset(root_dir):
    return parse.read_genesets(Path(root_dir, 'data/geneset-sample0.tsv').as_posix())

@pytest.fixture(scope='module')
def sample_ontology(root_dir):
    return parse.read_ontologies(Path(root_dir, 'data/ontology-sample0.tsv').as_posix())

@pytest.fixture
def sample_inputs(
    sample_annotations,
    sample_edges,
    sample_geneset,
    sample_homology,
    sample_ontology
):
    return types.Inputs(
        sample_annotations, sample_edges, sample_geneset, sample_homology, sample_ontology
    )

@pytest.fixture
def sample_node_map():
    return {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}

@pytest.fixture
def sample_graph():
    g = nx.DiGraph()
    g.add_edges_from([
        (0, 1), (0, 3), (0, 5), (1, 0), (1, 2), (2, 0), (2, 1), (3, 1),
        (3, 4), (4, 1), (5, 1), (5, 3)
    ])
    return g

@pytest.fixture
def sample_matrix():
    return np.fromstring(
        '0 1 0 1 0 1 1 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 0',
        sep=' '
    ).reshape(6, 6)


def test_encapsulate_bioentities(sample_inputs):

    inputs = graph.encapsulate_bioentities(sample_inputs)

    assert inputs.annotations.term.tolist() == [
       types.BioEntity('GO:0048149', 'term'),
       types.BioEntity('GO:0048149', 'term'),
       types.BioEntity('GO:0048149', 'term'),
    ]
    assert inputs.annotations.gene.tolist() == [
        types.BioEntity('DRD2', 'gene'),
        types.BioEntity('OPRM1', 'gene'),
        types.BioEntity('HDAC2', 'gene'),
    ]
    assert inputs.edges.source.tolist() == [
        types.BioEntity('0', 'gene'),
        types.BioEntity('1', 'gene'),
        types.BioEntity('0', 'gene'),
    ]
    assert inputs.edges.sink.tolist() == [
        types.BioEntity('1', 'gene'),
        types.BioEntity('2', 'gene'),
        types.BioEntity('2', 'gene'),
    ]
    assert inputs.genesets.geneset.tolist() == [
        types.BioEntity('1', 'geneset'),
        types.BioEntity('1', 'geneset'),
        types.BioEntity('2', 'geneset'),
    ]
    assert inputs.genesets.gene.tolist() == [
        types.BioEntity('DRD2', 'gene'),
        types.BioEntity('OPRM1', 'gene'),
        types.BioEntity('HDAC2', 'gene'),
    ]
    assert inputs.homology.cluster.tolist() == [
        types.BioEntity('0', 'homology'),
        types.BioEntity('0', 'homology'),
        types.BioEntity('0', 'homology'),
    ]
    assert inputs.homology.gene.tolist() == [
        types.BioEntity('ENSG00000149295', 'gene'),
        types.BioEntity('ENSMUSG00000032259', 'gene'),
        types.BioEntity('ENSRNOG00000008428', 'gene'),
    ]
    assert inputs.ontologies.child.tolist() == [
        types.BioEntity('GO:0048149', 'term'),
        types.BioEntity('GO:0045471', 'term'),
        types.BioEntity('GO:0030534', 'term'),
    ]
    assert inputs.ontologies.parent.tolist() == [
        types.BioEntity('GO:0045471', 'term'),
        types.BioEntity('GO:0009636', 'term'),
        types.BioEntity('GO:0009636', 'term'),
    ]


def test_generate_node_uids(sample_inputs):

    #inputs = graph.encapsulate_bioentities(sample_inputs)
    #uids = graph.generate_node_uids(inputs)
    uids = graph.generate_node_uids(sample_inputs)

    assert types.BioEntity('GO:0048149', 'term') in uids
    assert types.BioEntity('GO:0045471', 'term') in uids
    assert types.BioEntity('GO:0030534', 'term') in uids
    assert types.BioEntity('GO:0009636', 'term') in uids

    assert types.BioEntity('0', 'gene') in uids
    assert types.BioEntity('1', 'gene') in uids
    assert types.BioEntity('2', 'gene') in uids
    assert types.BioEntity('DRD2', 'gene') in uids
    assert types.BioEntity('OPRM1', 'gene') in uids
    assert types.BioEntity('HDAC2', 'gene') in uids
    assert types.BioEntity('ENSG00000149295', 'gene') in uids
    assert types.BioEntity('ENSMUSG00000032259', 'gene') in uids
    assert types.BioEntity('ENSRNOG00000008428', 'gene') in uids

    assert types.BioEntity('1', 'geneset') in uids
    assert types.BioEntity('2', 'geneset') in uids

    assert types.BioEntity('0', 'homology') in uids

    assert len(list(uids.keys())) == 16


def test_build_heterogeneous_graph(sample_inputs):

    uids = graph.generate_node_uids(sample_inputs)
    hetnet = graph.build_heterogeneous_graph(uids, sample_inputs)

    ## lol
    assert hetnet


def test_graph_to_sparse_matrix(sample_graph, sample_matrix):

    assert np.array_equal(
        graph.graph_to_sparse_matrix(sample_graph).toarray(),
        dok_matrix(sample_matrix).toarray()
    )


def test_column_normalize_matrix(sample_matrix):

    assert np.array_equal(
        np.round(graph.column_normalize_matrix(csr_matrix(dok_matrix(sample_matrix))).toarray(), 1),
        np.array([
            [0.0, 0.3, 0.0, 0.3, 0.0, 0.3],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
        ])
    )

def test_build_matrix(sample_graph):

    assert np.array_equal(
        np.round(graph.build_matrix(sample_graph).toarray(), 1),
        np.array([
            [0.0, 0.3, 0.0, 0.3, 0.0, 0.3],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
        ])
    )
