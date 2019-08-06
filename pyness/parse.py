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


def _read_df(input: str) -> pd.DataFrame:
    """
    Read an input file into a dataframe

    arguments
        input: input filepath

    returns
        a dataframe
    """

    return pd.read_csv(input, sep='\t')


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
    The edge list can have any number of fields as long as the first two fields represent
    the source and sink nodes respectively.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    df = _read_df(input)
    df = df.rename(columns={df.columns[0]: 'source', df.columns[1]: 'sink'})

    return df


def read_annotations(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing ontology annotations.
    The annotation file can have any number of fields as long as the first two fields
    represent the ontology term and gene ID respectively.

    arguments
        input: input filepath

    returns
        a dataframe
    """


    df = _read_df(input)
    df = df.rename(columns={df.columns[0]: 'term', df.columns[1]: 'gene'})

    return df


def read_genesets(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing gene sets.
    The annotation file can have any number of fields as long as the first two fields
    represent the gene set ID and gene IDs respectively.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    df = _read_df(input)
    df = df.rename(columns={df.columns[0]: 'gsid', df.columns[1]: 'genes'})

    ## Genes don't need to be split
    if not df.genes.str.contains('|').any():
        return df

    ## If genes are concatenated, split them up
    df['genes'] = df.genes.str.split('|')

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


def read_homology(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing homology mappings.
    The homology file can have any number of fields as long as the first two fields
    represent the cluster ID and gene ID respectively.

    arguments
        input: input filepath

    returns
        a dataframe
    """


    df = _read_df(input)
    df = df.rename(columns={df.columns[0]: 'cluster', df.columns[1]: 'gene'})

    return df


def read_ontologies(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing ontology relationships.
    The ontology file can have any number of fields as long as the first two fields
    represent the child and parent terms respectively.

    arguments
        input: input filepath

    returns
        a dataframe
    """


    df = _read_df(input)
    df = df.rename(columns={df.columns[0]: 'child', df.columns[1]: 'parent'})

    return df


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
            read_homology(df) for df in args.homology
        ])

    if args.ontologies:
        ontologies = pd.concat([
            read_ontologies(df) for df in args.ontologies
        ])

    return Inputs(annotations, edges, genesets, homology, ontologies)
