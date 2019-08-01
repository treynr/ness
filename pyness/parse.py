#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: parse.py
## desc: File parsing.
## auth: TR

from collections import namedtuple
from typing import Dict
import argparse
import logging
import pandas as pd

from . import log

logging.getLogger(__name__).addHandler(logging.NullHandler())


def read_seeds(input: str) -> pd.DataFrame:
    """
    Read and parse a seed list file.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    return pd.read_csv(input, sep='\t', header=None).iloc[:, 0].tolist()


def read_edges(input: str) -> pd.DataFrame:
    """
    Read and parse an edge list file.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    return pd.read_csv(input, sep='\t')


def read_annotations(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing ontology annotations.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    return pd.read_csv(input, sep='\t')


def read_genesets(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing gene sets.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    df = pd.read_csv(input, sep='\t')

    ## If genes are concatenated, split them up
    df['genes'] = df.genes.split('|')

    ## Identify fields that aren't the genes field
    id_vars = [c for c in df.columns if c != 'genes']

    ## Concat genes so each one has their own separate column
    df = pd.concat([
        df.drop(columns='genes'),
        df.genes.apply(pd.Series)
    ], axis=1)

    ## Melt the frame so there is one gene per row
    df = df.melt(id_vars=id_vars, value_name='gene')

    return df.drop(columns='variable').dropna()


def read_inputs(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    """
    Read and parse NESS inputs.

    arguments
        args: argument namespace

    returns
        a dict of NESS inputs
    """

    log._logger.info('Reading and parsing inputs...')

    Inputs = namedtuple(
        'Inputs', 'annotations edges genesets homology ontologies', defaults=(None,) * 5
    )
    annotations = None
    edges = None
    genesets = None
    homology = None
    ontologies = None

    if args.annotations:
        annotations = pd.concat([
            read_annotations(df) for df in args.annotations
        ])

    if args.edges:
        edges = pd.concat([
            read_edges(df) for df in args.edges
        ])

    if args.genesets:
        genesets = pd.concat([
            read_genesets(df) for df in args.genesets
        ])

    if args.homology:
        homology = pd.concat([
            read_edges(df) for df in args.homology
        ])

    if args.ontologies:
        ontologies = pd.concat([
            read_edges(df) for df in args.ontologies
        ])

    return Inputs(annotations, edges, genesets, homology, ontologies)
