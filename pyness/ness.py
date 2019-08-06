#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: ness.py
## desc: NESS python implementation.
## auth: TR

from dask.distributed import Client
from dask.distributed import LocalCluster
from dask.distributed import get_client
from functools import partial
from pathlib import Path
from scipy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from typing import Dict
from typing import List
import logging
import numpy as np
import networkx as nx
import pandas as pd
import os
import tempfile as tf

from . import graph
from . import log

logging.getLogger(__name__).addHandler(logging.NullHandler())


def merge_files(files: List[str], output: str, delete: bool = True) -> None:
    """

    :param arr:
    :param dex:
    :param value:
    :return:
    """

    if not files:
        return output

    first = True

    ## Open the single concatenated output file
    with open(output, 'w') as outfl:

        ## Loop through input files...
        for fpath in sorted(files):

            ## Read each input file and format line x line
            with open(fpath, 'r') as infl:

                if not first:
                    ## Skip the header
                    next(infl)

                else:
                    first = False

                    outfl.write(infl.read())

            ## Remove the file once we're done
            if delete:
                Path(fpath).unlink()


def _map_seed_uids(seeds: List[str], uids: Dict[str, int]) -> List[int]:
    """
    Map seed nodes to their UIDs.

    arguments
        seeds: list of seed nodes
        uids:  UID map

    returns
        a list of seed UIDs
    """

    seed_uids = []

    for s in seeds:
        if s not in uids:
            log._logger.warning(f'Skipping seed node {s} which is missing from the graph')
            continue

        seed_uids.append(uids[s])

    return seed_uids


def initialize_proximity_vector(size: int, seeds: List[int]) -> np.array:
    """
    Generate the initial proximity vector based on the given seeds.

    arguments
        size:  size of the proximity vector
        seeds: seed indexes for the vector

    returns
        the proximity vector
    """

    ## Generate a vector of zeros and set the seeds
    vector = np.zeros(size, dtype=np.double)
    vector[seeds] = 1.0

    ## If there are multiple seeds then equally weight them
    return (vector / vector.sum())


def calculate_proximity_vector(matrix, pvec, uvec, alpha):
    """
    Calculate the proximity vector. Follows the given equation:

        pu = (1 - a) A pu + a eu

        where a is the alpha (restart probability), pu is the proximity vector
        of node u, A is the column-stochastic transition probability matrix,
        and eu is the seed unit vector.

    arguments
        matrix: graph matrix
        pvec:   proximity vector
        uvec:   unit vector
        alpha:  restart probability

    returns
        the proximity vector
    """

    return ((1 - alpha) * matrix * pvec) + (alpha * uvec)


def calculate_convergence(v1, v2):
    """
    Calculate the l1 norm of the difference of two vectors.

    arguments
        v1: vector 1
        v2: vector 2

    returns
        the l1 norm of the vector difference
    """

    return norm(v2 - v1, ord=1)


def random_walk(matrix, seeds, alpha, threshold=1.0e-8):
    """
    Perform the random walk over the given transition probability matrix and seed values.

    arguments
        matrix:    graph matrix
        seeds:     seed indexes
        alpha:     restart prob.
        threshold: convergence threshold

    returns
        proximity vector from seed nodes to all other nodes
    """

    ## Calculate the initial vectors: the seed unit vector and the first proximity vector
    uvec = initialize_proximity_vector(matrix.shape[1], seeds)
    pre_vector = initialize_proximity_vector(matrix.shape[1], seeds)

    while True:

        ## Calculate the newest proximity vector
        current_vector = calculate_proximity_vector(matrix, pre_vector, uvec, alpha)

        ## Check if we've achieved convergence
        if calculate_convergence(pre_vector, current_vector) < threshold:
            break

        pre_vector = current_vector

    return current_vector


def _run_individual_walks(
    matrix: csr_matrix,
    seeds: List[str],
    uids: Dict[str, int],
    alpha: np.double = 0.15
) -> pd.DataFrame:
    """
    Run the random walk algorithm. Helper function for run_individual_walks.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
    """

    ## Reverse UID mapping
    reverse_uids = dict([(b, a) for a, b in uids.items()])

    ## Map seed UIDs
    mapped_seeds = _map_seed_uids(seeds, uids)

    prox_vectors = []

    for s in mapped_seeds:

        ## Walk the graph
        prox_vector = random_walk(matrix, [s], alpha)

        ## Construct the output
        prox_vector = pd.DataFrame(prox_vector, columns=['probability'])
        prox_vector = prox_vector.reset_index(drop=False)

        ## Map bio-entities back from UIDs
        prox_vector['node_to'] = prox_vector['index'].map(reverse_uids)

        ## Add the seed node
        prox_vector['node_from'] = reverse_uids[s]

        prox_vectors.append(prox_vector)

    return pd.concat(prox_vectors)


def run_individual_walks(
    matrix: csr_matrix,
    seeds: List[str],
    uids: Dict[str, int],
    output: str,
    alpha: np.double = 0.15
) -> None:
    """
    Run the random walk algorithm.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
    """

    ## Clear the output file if it exists
    with open(output, 'w') as fl:
        print('\t'.join(['node_from', 'node_to', 'probability']), file=fl)

    for s in seeds:

        prox_vector = _run_individual_walks(matrix, [s], uids, alpha)

        ## Save the output
        prox_vector.sort_values(by='probability', ascending=False).to_csv(
            output,
            mode='a',
            sep='\t',
            index=False,
            header=False,
            columns=['node_from', 'node_to', 'probability']
        )


def distribute_individual_walks(
    matrix: csr_matrix,
    seeds: List[str],
    uids: Dict[str, int],
    output: str,
    alpha: np.double = 0.15
) -> None:
    """
    Run the random walk algorithm.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
    """

    log._logger.info('Starting local cluster...')

    ## Start a local cluster utilizing all available cores
    client = Client(LocalCluster(n_workers=os.cpu_count()))

    log._logger.info('Scattering data to workers...')

    ## Scatter data onto workers
    matrix = client.scatter(matrix, broadcast=True)
    uids = client.scatter(uids, broadcast=True)

    futures = []

    log._logger.info('Walking the graph...')

    ## Split the seed list into chunks
    for chunk in np.array_split(seeds, os.cpu_count()):

        ## Temp output
        tmp_out = tf.NamedTemporaryFile().name

        ## Run the random walk algorithm for each seed
        future = client.submit(
            run_individual_walks,
            matrix,
            chunk,
            uids,
            tmp_out,
            alpha
        )

        futures.append(future)

    futures = client.gather(futures)

    log._logger.info('Generating output...')

    merge_files(futures, output)

    client.close()


def _run_individual_permutation_tests(
    matrix: csr_matrix,
    seed: List[str],
    uids: Dict[str, int],
    permutations: int = 250,
    alpha: np.double = 0.15
) -> pd.DataFrame:
    """
    Run permutation tests for individual random walks.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
    """

    log._logger.info('Running random walk...')

    ## First get the proximity vector for the walk
    prox_vector = _run_individual_walks(matrix, seed, uids, alpha)

    log._logger.info('Running permutation tests...')

    ## Start the permutation testing
    for i in range(permutations):

        ## Shuffle the node labels
        permuted_uids = graph.shuffle_node_labels(uids)

        ## Run the permuted walk
        permuted_vector = _run_individual_walks(matrix, seed, permuted_uids, alpha)

        ## Join on the original results
        prox_vector[f'p_{i}'] = permuted_vector.probability

    return prox_vector


def run_individual_permutation_tests(
    matrix: csr_matrix,
    seeds: List[str],
    uids: Dict[str, int],
    output: str,
    permutations: int = 250,
    alpha: np.double = 0.15
) -> None:
    """
    Run permutation tests for individual random walks.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
    """

    ## Clear the output file if it exists
    with open(output, 'w') as fl:
        print('\t'.join(['node_from', 'node_to', 'probability', 'p']), file=fl)

    for s in seeds:

        prox_vector = _run_individual_permutation_tests(
            matrix, [s], uids, permutations, alpha
        )

        log._logger.info('Calculating p-values...')

        ## Calculate the p-value using the cumulative probability of observing a walk
        ## score of equal or greater magnitude
        prox_vector['p'] = (
            prox_vector.filter(regex='p_\d+')
                .apply(lambda x: x >= prox_vector.probability)
                .select_dtypes(include=['bool'])
                .sum(axis=1)
        )
        prox_vector['p'] = (prox_vector['p'] + 1) / (permutations + 1)

        prox_vector = prox_vector[['node_from', 'node_to', 'probability', 'p']]

        ## Save the output
        prox_vector.sort_values(by='p', ascending=True).to_csv(
            output,
            mode='a',
            sep='\t',
            index=False,
            header=False,
            columns=['node_from', 'node_to', 'probability', 'p']
        )


def distribute_individual_permutation_tests(
    matrix: csr_matrix,
    seeds: List[str],
    uids: Dict[str, int],
    output: str,
    permutations: int = 250,
    alpha: np.double = 0.15
) -> None:
    """
    Run permutation tests for individual random walks.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
    """

    ## Clear the output file if it exists
    with open(output, 'w') as fl:
        print('\t'.join(['node_from', 'node_to', 'probability', 'p']), file=fl)

    client = get_client()

    log._logger.info('Scattering data to workers...')

    ## Scatter data onto workers
    matrix = client.scatter(matrix, broadcast=True)
    uids = client.scatter(uids, broadcast=True)

    futures = []

    for s in seeds:

        log._logger.info('Running permutation tests...')

        permuted_futures = []
        s = client.scatter([s], broadcast=True)

        ## Split the number of permutations evenly
        for chunk in np.array_split(np.zeros(permutations), os.cpu_count()):

            prox_vector_future = client.submit(
                _run_individual_permutation_tests,
                matrix,
                s,
                uids,
                len(chunk),
                alpha,
                pure=False
            )

            permuted_futures.append(prox_vector_future)

        futures.append(permuted_futures)

    log._logger.info('Calculating p-values...')

    ## Wait for testing to finish
    for test in futures:

        test = client.gather(test)
        prox_vector = test.pop(0)

        #print('wut')
        #print('wut')
        #print('wut')
        #print(prox_vector.head())
        #exit()
        for df in test:

            prox_vector = pd.concat([
                prox_vector,
                df.drop(columns=['node_from', 'node_to', 'probability'])
            ], axis=1)

        ## Calculate the p-value using the cumulative probability of observing a walk
        ## score of equal or greater magnitude
        prox_vector['p'] = (
            prox_vector.filter(regex='p_\d+')
                .apply(lambda x: x >= prox_vector.probability)
                .select_dtypes(include=['bool'])
                .sum(axis=1)
        )
        prox_vector['p'] = (prox_vector['p'] + 1) / (permutations + 1)

        prox_vector = prox_vector[['node_from', 'node_to', 'probability', 'p']]

        ## Save the output
        prox_vector.sort_values(by='p', ascending=True).to_csv(
            output,
            mode='a',
            sep='\t',
            index=False,
            header=False,
            columns=['node_from', 'node_to', 'probability', 'p']
        )

    client.close()


if __name__ == '__main__':
    from . import graph

    matrix = dok_matrix([
        #[0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0]
    ], dtype=np.double)

    matrix = dok_matrix([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0]
    ], dtype=np.double)

    #matrix[:, 0] = matrix[:, 0].todense() + 1
    #matrix = column_normalize_matrix(matrix)
    matrix = matrix.tocsr()
    matrix = graph.column_normalize_matrix(matrix)
    print(matrix.todense())
    exit()

    #print(calculate_convergence(
    #    initialize_proximity_vector(6, [0]),
    #    vec
    #))
    print(random_walk(matrix, [0], 0.15))
    print(random_walk(matrix, [1], 0.15))

    print('Generating graph...')

    graph = nx.fast_gnp_random_graph(2000, 0.3)

    print('Converting to matrix...')

    matrix = graph_to_matrix(graph)
    matrix = column_normalize_matrix(matrix)
    matrix = csr_matrix(matrix)

    matrix.eliminate_zeros()

    print(matrix[:,0])

    print('Walking graph...')

    vec = random_walk(matrix, [0], 0.15)
    vec2 = random_walk(matrix, [1], 0.15)

    print(vec[vec > 0])
    print(vec2[vec2 > 0])
    #print(list(graph.nodes)[:10])
