#!/usr/bin/env python
# -*- coding: utf8 -*-

## file: test_ness.py
## desc: Test functions in ness.py.

from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ness import graph
from ness import ness
from ness import types

@pytest.fixture
def sample_node_map():
    return {
        types.BioEntity('a'): 0,
        types.BioEntity('b'): 1,
        types.BioEntity('c'): 2,
        types.BioEntity('d'): 3,
        types.BioEntity('e'): 4,
        types.BioEntity('f'): 5
    }

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

@pytest.fixture
def sample_normalized_matrix(sample_matrix):
    return graph.column_normalize_matrix(csr_matrix(dok_matrix(sample_matrix)))


def test_random_walk(sample_node_map, sample_normalized_matrix):

    pvec = ness._run_individual_walks(
        sample_normalized_matrix,
        [types.BioEntity('b')],
        sample_node_map
    )

    assert np.round(pvec[pvec.node_to == 'a'].probability.iloc[0], 3) == 0.283
    assert np.round(pvec[pvec.node_to == 'b'].probability.iloc[0], 3) == 0.392
    assert np.round(pvec[pvec.node_to == 'c'].probability.iloc[0], 3) == 0.287
    assert np.round(pvec[pvec.node_to == 'd'].probability.iloc[0], 3) == 0.308
    assert np.round(pvec[pvec.node_to == 'e'].probability.iloc[0], 3) == 0.333
    assert np.round(pvec[pvec.node_to == 'f'].probability.iloc[0], 3) == 0.298
