#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: types.py
## desc: Data types.
## auth: TR

from dataclasses import dataclass
from dataclasses import field
from typing import Tuple
import pandas as pd

@dataclass(repr=True, eq=True, order=True, frozen=True)
class BioEntity:
    id: str
    biotype: str = 'default'
    # metadata: Dict[str, str] = field(default_factory=dict)

    def __str__(self):
        """
        To string.

        returns
            a string of the given bio ent.
        """

        return f'{self.biotype}:{self.id}'

    def has_id(self, s: str):
        return self.id == s


@dataclass(repr=True, eq=True, order=True, frozen=True)
class Gene(BioEntity):
    biotype: str = 'gene'


@dataclass(repr=True, eq=True, order=True, frozen=True)
class GeneSet(BioEntity):
    biotype: str = 'geneset'


@dataclass(repr=True, eq=True, order=True, frozen=True)
class Homolog(BioEntity):
    biotype: str = 'homolog'


@dataclass(repr=True, eq=True, order=True, frozen=True)
class Term(BioEntity):
    biotype: str = 'term'

@dataclass
class Options:
    annotations: Tuple[str]
    edges: Tuple[str]
    genesets: Tuple[str]
    homology: Tuple[str]
    ontologies: Tuple[str]
    seeds: str
    seed_file: str
    multiple: bool
    distributed: bool
    cores: int
    permutations: int
    restart: float
    verbose: bool
    output: str

@dataclass
class Inputs:
    annotations: pd.DataFrame = field(default=None)
    edges: pd.DataFrame = field(default=None)
    genesets: pd.DataFrame = field(default=None)
    homology: pd.DataFrame = field(default=None)
    ontologies: pd.DataFrame = field(default=None)

