#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: types.py
## desc: Data types.
## auth: TR

from dataclasses import dataclass

@dataclass(repr=True, eq=True, order=True, frozen=True)
class Gene:
    id: str

@dataclass(repr=True, eq=True, order=True, frozen=True)
class GeneSet:
    id: str

@dataclass(repr=True, eq=True, order=True, frozen=True)
class Homolog:
    id: str

@dataclass(repr=True, eq=True, order=True, frozen=True)
class Term:
    id: str

