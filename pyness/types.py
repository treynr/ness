#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: types.py
## desc: Data types.
## auth: TR

from dataclasses import dataclass

@dataclass(repr=True, eq=True, order=True, frozen=True)
class BioEntity:
    id: str
    biotype: str = ''

    def __str__(self):
        """
        To string.

        returns
            a string of the given bio ent.
        """

        return f'{self.biotype}:{self.id}'


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

