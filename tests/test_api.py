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

from ness import api

@pytest.fixture
def sample_df_0():
    return pd.DataFrame([['a', 'x'], ['b', 'y'], ['a', 'z']], columns=['a', 'b'])

@pytest.fixture
def sample_df_1():
    return pd.DataFrame([['a', 'x', 1], ['b', 'y', 2], ['a', 'z', 3]], columns=['a', 'b', 's'])

def test_format_input_dataframes_0(sample_df_0):
    assert api._format_input_dataframes(sample_df_0, ('b1', 'b2')).equals(pd.DataFrame([
        ['a', 'x', 'unknown_biotype', 'unknown_biotype', 1.0],
        ['b', 'y', 'unknown_biotype', 'unknown_biotype', 1.0],
        ['a', 'z', 'unknown_biotype', 'unknown_biotype', 1.0],
    ], columns=['b1', 'b2', 'biotype1', 'biotype2', 'score']))


def test_format_input_dataframes_1(sample_df_1):
    df = api._format_input_dataframes(sample_df_1, ('b1', 'b2'))
    df = df[['b1', 'b2', 'biotype1', 'biotype2', 'score']]

    assert df.equals(pd.DataFrame([
        ['a', 'x', 'unknown_biotype', 'unknown_biotype', 1],
        ['b', 'y', 'unknown_biotype', 'unknown_biotype', 2],
        ['a', 'z', 'unknown_biotype', 'unknown_biotype', 3],
    ], columns=['b1', 'b2', 'biotype1', 'biotype2', 'score']))

