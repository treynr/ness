#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: api.py
## desc: Functions designed to simplify graph building, matrix processing, and executing
##       the random walk when using NESS as a python library. Exposes a simple set of
##       API functions that can be called external Python code.
## auth: TR

from typing import List
from typing import Tuple
import logging
import pandas as pd

from . import graph
from .types import BioEntity
from .types import Inputs

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _format_input_dataframes(df: pd.DataFrame, columns: Tuple[str, str]) -> pd.DataFrame:
    """
    Makes an educated guess as to which columns specify which kind of data and fills
    in columns that are missing.

    arguments
        df:      dataframe containing bioentity data
        columns: column names to use for specifying each entity (there should only be two)

    returns
        a processed dataframe
    """

    ## This should never happen...
    if len(columns) != 2:
        raise ValueError(
            'ness.api._format_input_dataframes: columns requires exactly two elements'
        )

    ## Enforce column naming for the columns containing entity IDs
    df = df.rename(columns={df.columns[0]: columns[0], df.columns[1]: columns[1]})

    if len(df.columns) == 2:
        df['biotype1'] = 'unknown_biotype'
        df['biotype2'] = 'unknown_biotype'
        df['score'] = 1.0

    elif len(df.columns) == 3:
        df = df.rename(columns={df.columns[2]: 'score'})

        df['biotype1'] = 'unknown_biotype'
        df['biotype2'] = 'unknown_biotype'

    elif len(df.columns) == 4:
        df = df.rename(columns={df.columns[2]: 'biotype1', df.columns[3]: 'biotype2'})

        df['score'] = 1.0

    else:  # columns >= 5
        df = df.rename(columns={
            df.columns[2]: 'biotype1', df.columns[3]: 'biotype2', df.columns[4]: 'score'
        })

    return df


def format_inputs(
    annotations: pd.DataFrame = None,
    edges: pd.DataFrame = None,
    genesets: pd.DataFrame = None,
    homology: pd.DataFrame = None,
    ontologies: pd.DataFrame = None,
) -> Inputs:
    """
    Expects dataframes which have already been formatted and contain the correct amount
    of columns. Each DF should contain at least two columns representing the entity pairs
    which are linked in the graph.
    The first two columns in the DF are always entity 1 and entity 2. The next two columns
    should be the biotypes associated with E1 and E2. The last column is the score for
    that given edge.

    :param annotations:
    :param edges:
    :param genesets:
    :param homology:
    :param ontologies:
    :return:
    """

    inputs = Inputs()

    if annotations is not None:
        inputs.annotations = _format_input_dataframes(annotations, ('term', 'gene'))

    if edges is not None:
        inputs.edges = _format_input_dataframes(edges, ('source', 'sink'))

    if genesets is not None:
        inputs.genesets = _format_input_dataframes(genesets, ('geneset', 'gene'))

    if homology is not None:
        inputs.homology = _format_input_dataframes(homology, ('cluster', 'gene'))

    if ontologies is not None:
        inputs.ontologies = _format_input_dataframes(ontologies, ('child', 'parent'))

    return inputs


def format_seeds(seeds: pd.DataFrame, biotype: str = None) -> List[BioEntity]:
    """
    Formats seed nodes. The dataframe containing seeds should be at least one column which
    contains the entity identifier for the given seed. An optional column providing its
    biotype can be used. Alternatively, the biotype argument can be provided which will
    override and set biotypes for all seeds.

    arguments
        seeds:   dataframe of seed nodes
        biotype: optional biotype to use for all seeds

    returns

    """

    new_seeds = [] # type: ignore

    if seeds.empty:
        raise ValueError('ness.api._format_seeds: the seeds dataframe cannot be empty')

    if biotype:
        ## Only keep the first column which should contain node IDs, then set the biotype
        seeds = seeds.iloc[:, 0:1]
        seeds['biotype'] = biotype

    if len(seeds.columns) >= 2:
        ## Convert to BioEntity types
        new_seeds = (
            seeds.iloc[:, 0:2]
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
                .tolist()
        )

    else:
        new_seeds = seeds.iloc[:, 0].apply(lambda r: BioEntity(r)).tolist()

    return new_seeds


def build_graph_matrix(inputs: Inputs) -> Tuple:
    """
    Builds the NESS graph and adjacency matrix using the given inputs.

    arguments
        inputs: NESS inputs

    returns
        a tuple containing
            1) a UID map (dict) for converting between NESS IDs and original bioentity IDs
            2) the NESS graph as a networkx digraph
            3) the adjacency matrix in Compressed Sparse Row (CSR) matrix form
    """

    uids, hetnet = graph.build_graph(inputs)
    matrix = graph.build_matrix(hetnet)

    return (uids, hetnet, matrix)
