#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: parse.py
## desc: File parsing.
## auth: TR

from typing import List
import csv
import logging
import pandas as pd

from . import log
from .types import BioEntity
from .types import Options
from .types import Inputs

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _read_df(input: str) -> pd.DataFrame:
    """
    Read an input file into a dataframe

    arguments
        input: input filepath

    returns
        a dataframe
    """

    return pd.read_csv(input, sep='\t', comment='#')


def _has_header(input: str) -> bool:
    """
    Uses the CSV Sniffer class to determine if the given input file has a header or not.

    arguments
        input: the input filepath

    returns
        true if the file has a header, false otherwise
    """

    return csv.Sniffer().has_header(open(input, 'r').read(8192))


def read_seeds(input: str) -> List[BioEntity]:
    """
    Read and parse a seed list file.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    seeds = pd.read_csv(input, sep='\t', comment='#', header=None, dtype='str')

    ## Assume the second column is the bioentity type associated with the seed
    if len(seeds.columns) > 1:
        return (
            seeds.iloc[:, 0:2]
                .dropna(subset=[0])
                .apply(lambda r: BioEntity(r[0], r[1]), axis=1)
                .tolist()
        )

    ## Assume there's just one row with seed nodes
    else:
        return (
            seeds.iloc[:, 0]
                .dropna()
                .map(lambda r: BioEntity(r[0]))
                .tolist()
        )


def read_edges(input: str) -> pd.DataFrame:
    """
    Read and parse an edge list file.
    The edge list can have any number of fields as long as the first two fields represent
    the source and sink nodes respectively.
    If there are columns in positions 3 and 4, these columns are treated as the bioentity
    types for source and sink nodes.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    df = pd.read_csv(
        input,
        sep='\t',
        comment='#',
        header='infer' if _has_header(input) else None,
        dtype=str
    )
    df = df.rename(columns={df.columns[0]: 'source', df.columns[1]: 'sink'})

    ## Rename if bioentity type columns are supplied otherwise create default types
    ## for these entities
    if len(df.columns) >= 3:
        df = df.rename(columns={df.columns[2]: 'biotype1'})
    else:
        df['biotype1'] = 'gene'

    if len(df.columns) >= 4:
        df = df.rename(columns={df.columns[3]: 'biotype2'})
    else:
        df['biotype2'] = 'gene'

    return df


def read_annotations(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing ontology annotations.
    The annotation file can have any number of fields as long as the first two fields
    represent the ontology term and gene ID respectively.
    If there are columns in positions 3 and 4, these columns are treated as the bioentity
    types for term and gene entities.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    df = pd.read_csv(
        input,
        sep='\t',
        comment='#',
        header='infer' if _has_header(input) else None,
        dtype=str
    )
    df = df.rename(columns={df.columns[0]: 'term', df.columns[1]: 'gene'})

    ## Rename if bioentity type columns are supplied otherwise create default types
    ## for these entities
    if len(df.columns) >= 3:
        df = df.rename(columns={df.columns[2]: 'biotype1'})
    else:
        df['biotype1'] = 'term'

    if len(df.columns) >= 4:
        df = df.rename(columns={df.columns[3]: 'biotype2'})
    else:
        df['biotype2'] = 'gene'

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

    df = pd.read_csv(
        input,
        sep='\t',
        comment='#',
        header='infer' if _has_header(input) else None,
        dtype=str
    )
    df = df.rename(columns={df.columns[0]: 'geneset', df.columns[1]: 'gene'})

    ## Rename if bioentity type columns are supplied otherwise create default types
    ## for these entities
    if len(df.columns) >= 3:
        df = df.rename(columns={df.columns[2]: 'biotype1'})
    else:
        df['biotype1'] = 'geneset'

    if len(df.columns) >= 4:
        df = df.rename(columns={df.columns[3]: 'biotype2'})
    else:
        df['biotype2'] = 'gene'

    ## Genes don't need to be split
    if not df.gene.str.contains('|', regex=False).any():
        return df

    ## If genes are concatenated, split them up
    df['gene'] = df.gene.str.split('|')

    ## Identify fields that aren't the genes field
    id_vars = [c for c in df.columns if c != 'gene']

    ## Concat genes so each one has their own separate column
    df = pd.concat([
        df.drop(columns='gene'),
        df.gene.apply(pd.Series)
    ], axis=1)

    ## Melt the frame so there is one gene per row
    df = df.melt(id_vars=id_vars, value_name='gene')

    return df.drop(columns='variable').dropna()


def read_homology(input: str) -> pd.DataFrame:
    """
    Read and parse a file containing homology mappings.
    The homology file can have any number of fields as long as the first two fields
    represent the cluster ID and gene ID respectively.
    If there are columns in positions 3 and 4, these columns are treated as the bioentity
    types for cluster and gene entities.

    arguments
        input: input filepath

    returns
        a dataframe
    """

    df = pd.read_csv(
        input,
        sep='\t',
        comment='#',
        header='infer' if _has_header(input) else None,
        dtype=str
    )
    df = df.rename(columns={df.columns[0]: 'cluster', df.columns[1]: 'gene'})

    ## Rename if bioentity type columns are supplied otherwise create default types
    ## for these entities
    if len(df.columns) >= 3:
        df = df.rename(columns={df.columns[2]: 'biotype1'})
    else:
        df['biotype1'] = 'homology'

    if len(df.columns) >= 4:
        df = df.rename(columns={df.columns[3]: 'biotype2'})
    else:
        df['biotype2'] = 'gene'

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

    df = pd.read_csv(
        input,
        sep='\t',
        comment='#',
        header='infer' if _has_header(input) else None,
        dtype=str
    )
    df = df.rename(columns={df.columns[0]: 'child', df.columns[1]: 'parent'})

    ## Rename if bioentity type columns are supplied otherwise create default types
    ## for these entities
    if len(df.columns) >= 3:
        df = df.rename(columns={df.columns[2]: 'biotype1'})
    else:
        df['biotype1'] = 'term'

    if len(df.columns) >= 4:
        df = df.rename(columns={df.columns[3]: 'biotype2'})
    else:
        df['biotype2'] = 'term'

    return df


def read_inputs(options: Options) -> Inputs:
    """
    Read and parse NESS inputs.

    arguments
        options: CLI options

    returns
        a dict of NESS inputs
    """

    log._logger.info('Reading and parsing inputs...')

    inputs = Inputs()

    if options.annotations:
        inputs.annotations = pd.concat([
            read_annotations(df) for df in options.annotations
        ])

    if options.edges:
        inputs.edges = pd.concat([
            read_edges(df) for df in options.edges
        ])

    if options.genesets:
        inputs.genesets = pd.concat([
            read_genesets(df) for df in options.genesets
        ])

    if options.homology:
        inputs.homology = pd.concat([
            read_homology(df) for df in options.homology
        ])

    if options.ontologies:
        inputs.ontologies = pd.concat([
            read_ontologies(df) for df in options.ontologies
        ])

    return inputs

