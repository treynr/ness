#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: ness.py
## desc: NESS python implementation.
## auth: TR

from dask.distributed import get_client
from math import floor
from pathlib import Path
from scipy.linalg import norm
from scipy.sparse import csr_matrix
from typing import Dict
from typing import List
import logging
import numpy as np
import pandas as pd
import os
import tempfile as tf

from . import graph
from . import log
from . import types

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _make_ness_header_output(output: str, p: bool = False, q: bool = False) -> None:
    """
    Write the NESS header output to a file.

    arguments
        output: output filepath
    """

    columns = ['node_from', 'from_biotype', 'node_to', 'to_biotype', 'probability']

    if p:
        columns.append('p')
    if q:
        columns.append('q')

    with open(output, 'w') as fl:
        print('\t'.join(columns), file=fl)


def _append_ness_output(output: str, vector: pd.DataFrame) -> None:
    """
    Append NESS output to the given file.

    arguments
        output: output filepath
        vector: proximity vector results
    """

    columns = ['node_from', 'from_biotype', 'node_to', 'to_biotype', 'probability']

    if 'q' in vector.columns:
        columns.extend(['p', 'q'])
        vector = vector.sort_values(by='q', ascending=True)

    elif 'p' in vector.columns:
        columns.append('p')
        vector = vector.sort_values(by='p', ascending=True)

    else:
        vector = vector.sort_values(by='probability', ascending=False)

    vector.to_csv(
        output,
        mode='a',
        sep='\t',
        index=False,
        header=False,
        columns=columns
    )


def _merge_files(files: List[str], output: str, delete: bool = True) -> None:
    """
    Concatenate multiple files into a single file.

    arguments
        files:  list of files to concatenate
        output: output filepath
        delete: if true, delete individual files after adding their contents to the
                final output filepath
    """

    if not files:
        return

    first = True

    ## Open the single concatenated output file
    with open(output, 'w') as outfl:

        ## Loop through input files...
        for fpath in files:

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


def _map_seed_uids(seeds: List[types.BioEntity], uids: Dict[types.BioEntity, int]) -> List[int]:
    """
    Map seed nodes to their UIDs.

    arguments
        seeds: list of seed nodes
        uids:  UID map

    returns
        a list of seed UIDs
    """

    seed_uids = []
    all_biotypes = list(set([b.biotype for b in uids.keys()]))

    for s in seeds:
        ## If the user did not specify a seed biotype, then we must examine all biotypes
        ## for potential seed matches.
        if s.biotype == types._default_biotype:
            log._logger.warning((
                f'Seed node {s} is missing a biotype, '
                'searching all biotypes for matches...'
            ))

            for bt in all_biotypes:
                bioent = types.BioEntity(s.id, bt)

                if bioent in uids:
                    log._logger.warning(
                        f'Seed node {s} has a potential biotype match: {bt}'
                    )
                    seed_uids.append(uids[bioent])
            continue

        if s not in uids:
            log._logger.warning(f'Skipping seed node {s} which is missing from the graph')
            continue

        seed_uids.append(uids[s])

    if not seed_uids:
        log._logger.warning('No seed nodes were mapped to UIDs, cannot perform the RWR')

    return seed_uids


def _calculate_p(vector: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Calculate p-values for a permuted walk dataset. The p-value is the cumulative
    probability of observing a walk score of equal or greater magnitude.

    arguments
        vector: proximity vector results
        n:      the number of permutations

    returns
        a proximity vector with p-values attached
    """

    ## Isolate permuted walk scores (these fields begin w/ p_), identify permuted scores
    ## of equal or greater magnitude than the one originally observed, sum the scores,
    ## then calculate the p-value.
    vector['p'] = (
        vector.filter(regex=r'p_\d+')
            .apply(lambda x: x >= vector.probability)
            .select_dtypes(include=['bool'])
            .sum(axis=1)
    )
    vector['p'] = (vector.p + 1) / (n + 1)

    ## Get rid of all the permuted score columns
    return vector[['node_from', 'node_to', 'probability', 'p']]


def _adjust_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce FDR adjusted p-values for the given dataset.

    arguments
        df: dataframe containing walk scores and p-values

    returns
        a dataframe with adjust p-values
    """

    df = df.sort_values(by='p').reset_index(drop=True)
    df['q'] = df.p * len(df.index) / (df.index + 1)
    df['q'] = df.q.mask(df.q > 1.0, 1.0)

    return df


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
    seeds: List[types.BioEntity],
    uids: Dict[types.BioEntity, int],
    alpha: np.double = 0.15,
    single: bool = False
) -> pd.DataFrame:
    """
    Run the random walk algorithm. Helper function called by the run_* and distribute_*
    functions.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
        single: run a single walk startng from all seed nodes at once
    """

    ## Reverse UID -> bioentity mapping
    revuids = dict([(b, a) for a, b in uids.items()])

    ## Map seed UIDs
    if single:
        mapped_seeds = _map_seed_uids(seeds[0], uids) # type: ignore
    else:
        mapped_seeds = _map_seed_uids(seeds, uids)

    if not mapped_seeds:
        return None

    ## Walk the graph
    prox_vector = random_walk(matrix, mapped_seeds, alpha)

    ## Construct the output
    prox_vector = pd.DataFrame(prox_vector, columns=['probability'])
    prox_vector = prox_vector.reset_index(drop=False)

    ## Map bio-entities back from UIDs
    prox_vector[['node_to', 'to_biotype']] = (pd.DataFrame(
        prox_vector['index']
            .map(revuids)
            .map(lambda b: [b.id, b.biotype])
            .tolist()
    ))

    ## Add the seed node
    prox_vector['node_from'] = ';'.join([revuids[s].id for s in mapped_seeds])
    prox_vector['from_biotype'] = ';'.join([revuids[s].biotype for s in mapped_seeds])

    return prox_vector


def run_individual_walks(
    matrix: csr_matrix,
    seeds: List[types.BioEntity],
    uids: Dict[types.BioEntity, int],
    output: str,
    alpha: np.double = 0.15,
    single: bool = False
) -> str:
    """
    Run the random walk algorithm for each seed in the given seeds list.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
        single: run a single walk startng from all seed nodes at once

    returns
        the output filepath
    """

    ## Clear the output file if it exists
    _make_ness_header_output(output)

    log._logger.info('Walking the graph...')

    ## Start from all seeds at once...
    if single:
        prox_vector = _run_individual_walks(matrix, seeds, uids, alpha)

        if prox_vector is not None:
            _append_ness_output(output, prox_vector)

    ## Perform separate walks for each individual seed
    else:
        for s in seeds:
            prox_vector = _run_individual_walks(matrix, [s], uids, alpha)

            if prox_vector is not None:
                _append_ness_output(output, prox_vector)

    return output


def distribute_individual_walks(
    matrix: csr_matrix,
    seeds: List[types.BioEntity],
    uids: Dict[types.BioEntity, int],
    output: str,
    alpha: np.double = 0.15,
    procs: int = os.cpu_count()
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

    client = get_client()

    ## Scatter data onto all the workers
    [matrix] = client.scatter([matrix], broadcast=True)
    [uids] = client.scatter([uids], broadcast=True)

    futures = []

    for chunk in np.array_split(seeds, floor(procs + (procs / 2))):

        if chunk.size == 0:
            continue

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

    _merge_files(futures, output)


def _run_individual_permutation_tests(
    matrix: csr_matrix,
    seeds: List[types.BioEntity],
    uids: Dict[types.BioEntity, int],
    permutations: int = 250,
    alpha: np.double = 0.15,
    single: bool = False
) -> pd.DataFrame:
    """
    Helper function for permutation testing functions. Runs the RWR algorithm and
    performs permutation testing over permuted graphs for the given seed nodes.

    arguments
        matrix: transition probability matrix
        seeds:  seed list
        uids:   UID map
        output: output filepath
        alpha:  restart probability
        single: perform a single random walk from all seed nodes at once
    """

    ## First get the proximity vector for the walk
    prox_vector = _run_individual_walks(matrix, seeds, uids, alpha, single=single)

    ## Start the permutation testing
    for i in range(permutations):

        ## Shuffle the node labels
        permuted_uids = graph.shuffle_node_labels(uids)

        ## Run the permuted walk
        permuted_vector = _run_individual_walks(
            matrix, seeds, permuted_uids, alpha, single=single
        )

        ## Join on the original results
        prox_vector[f'p_{i}'] = permuted_vector.probability

    return prox_vector


def run_individual_permutation_tests(
    matrix: csr_matrix,
    seeds: List[types.BioEntity],
    uids: Dict[types.BioEntity, int],
    output: str,
    permutations: int = 250,
    alpha: np.double = 0.15,
    single: bool = False
) -> None:
    """
    Run the random walk algorithm for each seed in the given seeds list and also perform
    permutation testing.
    This is NOT recommended for large inputs (or even for a single input). It is
    recommended to use the distributed version of this function when doing permutation
    testing.

    arguments
        matrix:       transition probability matrix
        seeds:        seed list
        uids:         UID map
        output:       output filepath
        permutations: number of permutations to run
        alpha:        restart probability
        single:     start from multiple seed nodes at once
    """

    ## Clear the output file if it exists
    _make_ness_header_output(output)

    ## Wraps the list of seeds in one extra list so the for loop below only iterates
    ## through on loop and the walk starts from all seeds
    if single:
        seeds = [seeds] # type: ignore

    for s in seeds:

        prox_vector = _run_individual_permutation_tests(
            matrix, [s], uids, permutations, alpha
        )

        ## Calculate the p-value
        prox_vector = _calculate_p(prox_vector, permutations)

        ## Save the output
        _append_ness_output(output, prox_vector)


def distribute_individual_permutation_tests(
    matrix: csr_matrix,
    seeds: List[types.BioEntity],
    uids: Dict[types.BioEntity, int],
    output: str,
    permutations: int = 250,
    alpha: np.double = 0.15,
    procs: int = os.cpu_count(),
    single: bool = False,
    fdr: bool = False
) -> None:
    """
    Run the random walk algorithm for seeds in the given seeds list and also perform
    permutation testing.
    Permutation tests are parallelized and distributed across a local cluster.

    arguments
        matrix:       transition probability matrix
        seeds:        seed list
        uids:         UID map
        output:       output filepath
        permutations: number of permutations to run
        alpha:        restart probability
        single:       start from multiple seed nodes at once
    """

    client = get_client()

    log._logger.info('Scattering data to workers...')

    ## Scatter data onto workers
    [matrix] = client.scatter([matrix], broadcast=True)
    [uids] = client.scatter([uids], broadcast=True)
    futures = []

    if single:
        seeds = [seeds] # type: ignore

    for s in seeds:

        log._logger.info('Running permutation tests...')

        permuted_futures = []
        s = client.scatter([s], broadcast=True)

        ## Split the number of permutations evenly
        for chunk in np.array_split(np.zeros(permutations), procs):

            prox_vector_future = client.submit(
                _run_individual_permutation_tests,
                matrix,
                s,
                uids,
                len(chunk),
                alpha,
                pure=False,
                single=single
            )

            permuted_futures.append(prox_vector_future)

        futures.append(permuted_futures)

    log._logger.info('Calculating p-values...')

    ## Wait for testing to finish
    for i, test in enumerate(futures):

        ## Gather the results of the permutation tests for this specific seed node
        test = client.gather(test)
        ## Get the first test so we keep the node_from, node_to, and prob. columns and
        ## concat the walk scores from the rest.
        prox_vector = test.pop(0)

        ## Get rid of node_from, node_to, prob. columns from the rest of the tests and
        ## only keep their permuted walk scores
        for df in test:
            prox_vector = pd.concat([
                prox_vector,
                df.drop(columns=['node_from', 'node_to', 'probability'])
            ], axis=1)

        ## Calculate the p-value
        prox_vector = _calculate_p(prox_vector, permutations)

        ## FDR adjusted p-values
        if fdr:
            prox_vector = _adjust_fdr(prox_vector)

        ## Create a new file if necessary
        if i == 0:
            _make_ness_header_output(output, p=True, q=fdr)

        ## Save the output
        _append_ness_output(output, prox_vector)

