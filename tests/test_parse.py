#!/usr/bin/env python
# -*- coding: utf8 -*-

## file: test_parse.py
## desc: Test functions in parse.py.

from pathlib import Path
from pyness import parse
from pyness import types
import pandas as pd
import pytest

@pytest.fixture(scope='module')
def root_dir():
    return Path(__file__).resolve().parent.as_posix()

@pytest.fixture(scope='module')
def sample_seeds_0(root_dir):
    return Path(root_dir, 'data/seed-sample0.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_seeds_1(root_dir):
    return Path(root_dir, 'data/seed-sample1.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_seeds_2(root_dir):
    return Path(root_dir, 'data/seed-sample2.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_edges_0(root_dir):
    return Path(root_dir, 'data/edge-sample0.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_edges_1(root_dir):
    return Path(root_dir, 'data/edge-sample1.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_annotations_0(root_dir):
    return Path(root_dir, 'data/annotation-sample0.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_annotations_1(root_dir):
    return Path(root_dir, 'data/annotation-sample1.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_homology_0(root_dir):
    return Path(root_dir, 'data/homology-sample0.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_homology_1(root_dir):
    return Path(root_dir, 'data/homology-sample1.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_geneset_0(root_dir):
    return Path(root_dir, 'data/geneset-sample0.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_geneset_1(root_dir):
    return Path(root_dir, 'data/geneset-sample1.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_ontology_0(root_dir):
    return Path(root_dir, 'data/ontology-sample0.tsv').as_posix()

@pytest.fixture(scope='module')
def sample_ontology_1(root_dir):
    return Path(root_dir, 'data/ontology-sample1.tsv').as_posix()


def test_seed_file_parsing_0(sample_seeds_0):

    seeds = parse.read_seeds(sample_seeds_0)

    assert seeds
    assert seeds[0] == types.BioEntity('1', 'node')
    assert seeds[1] == types.BioEntity('2', 'node')
    assert seeds[2] == types.BioEntity('3', 'vertex')


def test_seed_file_parsing_1(sample_seeds_1):

    seeds = parse.read_seeds(sample_seeds_1)

    assert seeds
    assert seeds[0] == types.BioEntity('1')
    assert seeds[1] == types.BioEntity('2')
    assert seeds[2] == types.BioEntity('3')


def test_seed_file_parsing_2(sample_seeds_2):

    seeds = parse.read_seeds(sample_seeds_2)

    assert seeds
    assert seeds[0] == types.BioEntity('id', 'class')
    assert seeds[1] == types.BioEntity('1', 'node')
    assert seeds[2] == types.BioEntity('2', 'node')
    assert seeds[3] == types.BioEntity('3', 'node')


def test_edge_file_parsing_0(sample_edges_0):

    edges = parse.read_edges(sample_edges_0)

    assert not edges.empty
    assert edges.equals(pd.DataFrame([
        ['0', '1', 'gene', 'gene'],
        ['1', '2', 'gene', 'gene'],
        ['0', '2', 'gene', 'gene']
    ], columns=['source', 'sink', 'biotype1', 'biotype2']))


def test_edge_file_parsing_1(sample_edges_1):

    edges = parse.read_edges(sample_edges_1)

    assert not edges.empty
    assert edges.equals(pd.DataFrame([
        ['0', '1', 'gene', 'idk'],
        ['1', '2', 'idk', 'gene'],
        ['0', '2', 'gene', 'gene']
    ], columns=['source', 'sink', 'biotype1', 'biotype2']))


def test_annotation_file_parsing_0(sample_annotations_0):

    annos = parse.read_annotations(sample_annotations_0)

    assert not annos.empty
    assert annos.equals(pd.DataFrame([
        ['GO:0048149', 'DRD2', 'term', 'gene'],
        ['GO:0048149', 'OPRM1', 'term', 'gene'],
        ['GO:0048149', 'HDAC2', 'term', 'gene']
    ], columns=['term', 'gene', 'biotype1', 'biotype2']))


def test_annotation_file_parsing_1(sample_annotations_1):

    annos = parse.read_annotations(sample_annotations_1)

    assert not annos.empty
    assert annos.equals(pd.DataFrame([
        ['GO:0048149', 'DRD2', 'go_term', 'gene'],
        ['GO:0048149', 'OPRM1', 'go_term', 'gene'],
        ['GO:0048149', 'HDAC2', 'go_term', 'gene']
    ], columns=['term', 'gene', 'biotype1', 'biotype2']))


def test_homology_file_parsing_0(sample_homology_0):

    homs = parse.read_homology(sample_homology_0)

    assert not homs.empty
    assert homs.equals(pd.DataFrame([
        ['0', 'ENSG00000149295', 'homology', 'gene'],
        ['0', 'ENSMUSG00000032259', 'homology', 'gene'],
        ['0', 'ENSRNOG00000008428', 'homology', 'gene']
    ], columns=['cluster', 'gene', 'biotype1', 'biotype2']))


def test_homology_file_parsing_1(sample_homology_1):

    homs = parse.read_homology(sample_homology_1)

    assert not homs.empty
    assert homs.equals(pd.DataFrame([
        ['0', 'ENSG00000149295', 'ortholog', 'gene'],
        ['0', 'ENSMUSG00000032259', 'ortholog', 'gene'],
        ['0', 'ENSRNOG00000008428', 'ortholog', 'gene']
    ], columns=['cluster', 'gene', 'biotype1', 'biotype2']))


def test_geneset_file_parsing_0(sample_geneset_0):

    gs = parse.read_genesets(sample_geneset_0)

    assert not gs.empty
    assert gs.equals(pd.DataFrame([
        ['1', 'DRD2', 'geneset', 'gene'],
        ['1', 'OPRM1', 'geneset', 'gene'],
        ['2', 'HDAC2', 'geneset', 'gene']
    ], columns=['geneset', 'gene', 'biotype1', 'biotype2']))


def test_geneset_file_parsing_1(sample_geneset_1):

    gs = parse.read_genesets(sample_geneset_1)

    print(gs)
    assert not gs.empty
    assert gs.equals(pd.DataFrame([
        ['1', 'geneset', 'gene', 'DRD2'],
        ['2', 'geneset', 'gene', 'HDAC2'],
        ['1', 'geneset', 'gene', 'OPRM1']
    ], columns=['geneset', 'biotype1', 'biotype2', 'gene']))


def test_ontology_file_parsing_0(sample_ontology_0):

    onts = parse.read_ontologies(sample_ontology_0)

    assert not onts.empty
    assert onts.equals(pd.DataFrame([
        ['GO:0048149', 'GO:0045471', 'term', 'term'],
        ['GO:0045471', 'GO:0009636', 'term', 'term'],
        ['GO:0030534', 'GO:0009636', 'term', 'term']
    ], columns=['child', 'parent', 'biotype1', 'biotype2']))


def test_ontology_file_parsing_1(sample_ontology_1):

    onts = parse.read_ontologies(sample_ontology_1)

    assert not onts.empty
    assert onts.equals(pd.DataFrame([
        ['GO:0048149', 'GO:0045471', 'go_term', 'go_term'],
        ['GO:0045471', 'GO:0009636', 'go_term', 'go_term'],
        ['GO:0030534', 'GO:0009636', 'go_term', 'go_term']
    ], columns=['child', 'parent', 'biotype1', 'biotype2']))

