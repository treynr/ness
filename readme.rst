
Network Enhanced Similarity Search (NESS)
=========================================

.. image:: https://img.shields.io/circleci/build/github/treynr/ness/master?style=flat-square&token=3e277067ea5de25755905e093e40d0e70db4c3cf
    :target: https://circleci.com/gh/treynr/ness

NESS aggregates and harmonizes heterogeneous graph types
including ontologies and their annotations, biological networks, and
bipartite representations of experimental study results across species.
The tool employs diffusion metrics, specifically a random walk with
restart (RWR), to estimate the relations among entities in the graph and
to make data-driven comparisons.


Usage
-----

.. code:: text

    Usage: ness [OPTIONS] [OUTPUT]

      Network Enhanced Similarity Search (NESS).
      Integrate heterogeneous functional genomics datasets and calculate
      diffusion metrics over the heterogeneous network using a random
      walk with restart.

    Options:
      -a, --annotations PATH      ontology annotation files
      -e, --edges PATH            edge list files
      -g, --genesets PATH         gene set files
      -h, --homology PATH         homology mapping files
      -o, --ontologies PATH       ontology relation files
      -s, --seeds TEXT            seed node
      --seed-file PATH            file containing list of seed nodes
      -m, --multiple              if using multiple seed nodes, start from all
                                  seeds at once instead of one walk per seed
      -d, --distributed           parallelize the random walk by distributing
                                  across available cores
      -c, --cores INTEGER         distribute computations among N cores (default =
                                  all available cores)
      -n, --permutations INTEGER  permutations to run
      -r, --restart FLOAT         restart probability
      --verbose                   clutter your screen with output
      --version                   Show the version and exit.
      --help                      Show this message and exit.

For example, NESS can be used calculate all pairwise gene affinities for genes in
biological network, with a restart probability of 0.15, and save the results to a file:

.. code:: text

    $ ness -e network.txt -r 0.15 results.tsv

Or, calculate the affinity between a single gene (e.g., MOBP) and all other genes in
the network:

.. code:: text

    $ ness -e network.txt -s MOBP results.tsv

For large networks, the random walk can be distributed across cores (eight in this case):

.. code:: text

    $ ness -e network.txt -d -c 8 results.tsv

The significance of any results can be assessed via permutations of the graph:

.. code:: text

    $ ness -e network.txt -p 500 results.tsv


Inputs
------

NESS accepts five different input types for contstructing the heterogeneous graph:

edge lists
    ``-e/--edges``. Undirected edges from biological networks (e.g., BioGRID__).

gene sets
    ``-g/--genesets``. Bipartite set-gene associations from gene set resources such as
    GeneWeaver__.

homology mappings
    ``-h/--homology``. Bipartite associations between homology clusters and genes.
    Direct associations between homologous genes in separate species can also be used.

ontologies
    ``-o/--ontologies``. Directed acyclic graphs (DAG) used for knowledge representation in
    ontologies such as the `Gene Ontology`__ (GO).

ontology annotations
    ``-a/--annotations``. Bipartite term-annotation associations from resources such as
    the GO or `Mammalian Phenotype Ontology`__.

.. __: https://thebiogrid.org/
.. __: https://geneweaver.org
.. __: http://geneontology.org/
.. __: http://www.informatics.jax.org/vocab/mp_ontology/

All input formats *must* be tab delimited.
Header rows are optional--NESS will attempt to detect if one is present or not.
Rows that begin with '#' are treated as comments.


Edge lists
''''''''''

Edge list inputs contain four columns, the last two are optional.
Each row lists a separate edge.
The first two columns specify identifiers for the two nodes that comprise the edge.
The last two (optional) columns describe the biotypes for the two nodes.

.. csv-table:: Example edge list input
    :header: node1, node2, biotype1, biotype2

    ENSG00000168314, ENSP00000312293, gene, protein
    ENSG00000168314, ENSG00000184221, gene, gene
    ENSG00000184221, ENSG00000205927, gene, gene


Gene sets
'''''''''

Gene set inputs contain four columns, the last two are optional.
Each row lists a single gene contained in a specific set.
The first two columns specify identifiers for the sets and genes respectively.
The last two (optional) columns describe the biotypes for the sets and genes.

.. csv-table:: Example gene set input
    :header: geneset, gene, set_biotype, gene_biotype

    Upregulated in opioid dependence, DRD2, geneset, gene
    Upregulated in opioid dependence, OPRM1, geneset, gene
    GS1337, HDAC1, geneset, gene
    GS1337, HDAC2, geneset, gene

Alternatively, gene sets can also be specified per row by separating genes using
the pipe '|' character.
The input above can be reformatted as:

.. csv-table:: Example gene set input
    :header: geneset, gene, set_biotype, gene_biotype

    Upregulated in opioid dependence, DRD2|OPRM1, geneset, gene
    GS1337, HDAC1|HDAC2, geneset, gene


Homology mappings
'''''''''''''''''

Homology inputs contain four columns, the last two are optional.
Each row lists a cluster of homologous genes and a gene belonging to that cluster.
The first two columns specify identifiers for the cluster and gene respectively.
The last two (optional) columns describe the biotypes for the cluster and gene.

.. csv-table:: Example homology input
    :header: cluster, gene, cluster_biotype, gene_biotype

    1, ENSG00000149295, ortholog, gene
    1, ENSMUSG00000032259, ortholog, gene
    1, ENSRNOG00000008428, ortholog, gene


Ontologies
''''''''''

Ontology inputs contain four columns, the last two are optional.
Each row lists a single term-term edge present in a DAG.
The first two columns specify identifiers for the ontology terms comprising the edge.
The last two (optional) columns describe the biotypes for the terms.

.. csv-table:: Example ontology input
    :header: term1, term2, biotype1, biotype2

    GO:0048149, GO:0045471, go_bp, go_bp
    GO:0045471, GO:0009636, go_bp, go_bp
    GO:0030534, GO:0009636, go_bp, go_bp


Ontology annotations
''''''''''''''''''''

Ontology annotation inputs contain four columns, the last two are optional.
Each row lists a single term annotation.
The first two columns specify identifiers for the ontology term and its annotation.
The last two (optional) columns describe the biotypes for the term and annotation.

.. csv-table:: Example annotation input
    :header: term, gene, term_biotype, gene_biotype

    GO:0048149, DRD2, go_bp, gene
    GO:0048149, OPRM1, go_bp, gene
    GO:0048149, HDAC2, go_bp, gene


Installation
------------

The current release is :code:`v1.1.0`.
Install via pip:

.. code:: bash

    $ pip install https://github.com/treynr/ness/releases/download/v1.1.0/ness-1.1.0.tar.gz

Or clone this repo and install using poetry__:

.. code:: bash

    $ git clone https://github.com/treynr/ness.git
    $ cd ness
    $ poetry install
    $ poetry run ness

.. __: https://python-poetry.org/


Testing
'''''''

Run unit and functional tests:

.. code:: bash

    $ PYTHONPATH=. pytest tests -v

..type checks:

.. code:: bash

    $ mypy --show-error-codes --ignore-missing-imports --no-strict-optional ness

..and style checks:

.. code:: bash

    $ flake8 ness


Requirements
------------

See ``pyproject.toml`` for a complete list of required Python packages.
The major requirements are:

- Python >= 3.7
- dask__
- networkx__
- numpy__
- pandas__
- scipy__

.. __: https://dask.org/
.. __: https://networkx.github.io/
.. __: https://numpy.org/
.. __: https://pandas.pydata.org/
.. __: https://scipy.org/


Funding
-------

Part of the GeneWeaver__ data repository and analysis platform.
For a detailed description of GeneWeaver, see this article__.

This work has been supported by joint funding from the NIAAA and NIDA, NIH [R01 AA18776];
and `The Jackson Laboratory`__ (JAX) Center for Precision Genetics of the
NIH [U54 OD020351].

.. __: https://geneweaver.org
.. __: https://www.ncbi.nlm.nih.gov/pubmed/26656951
.. __: https://jax.org/
