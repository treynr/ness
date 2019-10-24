#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: cli.py
## desc: CLI interface for NESS.
## auth: TR

import click
from dask.distributed import Client
from dask.distributed import LocalCluster
from functools import partial
from scipy.sparse import csr_matrix
from typing import Dict
from typing import List
from typing import Tuple
import logging
import os
import tempfile as tf

from . import __version__
# from . import arguments
from . import log
from . import graph
from . import ness
from . import parse
from . import types

logging.getLogger(__name__).addHandler(logging.NullHandler())


# def _wait_for_workers(n, min_n=0, limit=30, sleepy=5):
#
#     client = get_client()
#     waits = 0
#     num_started = 0
#
#     while True:
#         if limit > 0 and waits > limit:
#             log._logger.warning(
#                 'Waiting time limit has been reached, starting without additional workers'
#             )
#             break
#
#         workers =  client.scheduler_info()["workers"].keys()
#
#         sleep(sleepy)
#
#         if len(workers) > 0 and len(workers) != num_started:
#             num_started = len(workers)
#
#             log._logger.info(f'Started {len(workers)} workers...')
#
#         if len(workers) >= n:
#             log._logger.info(f'All {len(workers)} workers have started...')
#             break
#
#         if min_n > 0 and len(workers) >= min_n:
#             break
#
#         waits += 1


# def _main() -> None:
#     """
#     Application main.
#     """
#
#     args = arguments.setup_arguments('pyness')
#
#     log._initialize_logging(verbose=args.verbose)
#
#     if args.seeds:
#         seeds = args.seeds
#
#     elif args.seed_file:
#         seeds = parse.read_seeds(args.seed_file)
#
#     else:
#         log._logger.warning('You did not provide any seed nodes...')
#         log._logger.warning(
#             'This will calculate all pairwise probabilities which could take awhile'
#         )
#
#         seeds = []
#
#     #print(seeds[:5])
#     inputs = parse.read_inputs(args)
#     uids, hetnet = graph.build_graph(inputs)
#     matrix = graph.build_matrix(hetnet)
#     #print(list(uids.keys())[:5])
#
#     #for s in seeds:
#     #    if s not in uids:
#     #        print(f'{s} not in uid')
#
#     #for s in uids.keys():
#     #    if s.biotype == 'geneset':
#     #        print(s)
#
#     ## Using all nodes in the graph
#     if not seeds:
#         seeds = list(uids.keys())
#
#     if args.multiple:
#         if args.permutations:
#
#             ## Start a local cluster utilizing the specified number of cores
#             client = Client(LocalCluster(
#                 n_workers=args.cores,
#                 processes=True,
#                 local_directory=tf.gettempdir()
#             ))
#
#             ## Run the logging init function on each worker and register the callback so
#             ## future workers also run the function
#             init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
#             client.register_worker_callbacks(setup=init_logging_partial)
#
#             ## Do the permutation testing
#             ness.distribute_individual_permutation_tests(
#                 matrix,
#                 [seeds],
#                 uids,
#                 args.output,
#                 permutations=args.permutations,
#                 alpha=args.restart,
#                 fdr=args.fdr
#             )
#
#             client.close()
#
#         else:
#             if args.distributed:
#                 log._logger.warning((
#                     'Distributed computing is unnecessary when starting from '
#                     'multiple seeds'
#                 ))
#
#     else:
#         ## We're doing permutation testing...
#         if args.permutations:
#
#             ## Distribute permutation tests across a local cluster
#             if args.distributed:
#
#                 ## Start a local cluster utilizing the specified number of cores
#                 client = Client(LocalCluster(
#                     n_workers=args.cores,
#                     processes=True,
#                     local_directory=tf.gettempdir()
#                 ))
#
#                 ## Run the logging init function on each worker and register the callback so
#                 ## future workers also run the function
#                 init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
#                 client.register_worker_callbacks(setup=init_logging_partial)
#
#                 ## Do the permutation testing
#                 ness.distribute_individual_permutation_tests(
#                     matrix,
#                     seeds,
#                     uids,
#                     args.output,
#                     permutations=args.permutations,
#                     alpha=args.restart,
#                     procs=args.cores,
#                     fdr=args.fdr
#                 )
#
#                 client.close()
#
#             else:
#                 ## Run the permutation testing
#                 ness.run_individual_permutation_tests(
#                     matrix, seeds, uids, args.output, args.permutations, args.restart
#                 )
#
#         ## No permutation testing, just the walk...
#         else:
#             if args.distributed:
#
#                 log._logger.info(f'Starting local cluster: {args.cores} cores...')
#
#                 ## Start a local cluster utilizing the specified number of cores
#                 client = Client(LocalCluster(
#                     n_workers=args.cores,
#                     processes=True,
#                     local_directory=tf.gettempdir()
#                 ))
#
#                 log._logger.info(f'Started local cluster...')
#
#                 _wait_for_workers(args.cores, limit=0, sleepy=2)
#
#                 ## Run the logging init function on each worker and register the callback so
#                 ## future workers also run the function
#                 init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
#                 client.register_worker_callbacks(setup=init_logging_partial)
#
#                 log._logger.info(f'Walking the graph...')
#
#                 ness.distribute_individual_walks(
#                     matrix, seeds, uids, args.output, alpha=args.restart, procs=args.cores
#                 )
#
#                 client.close()
#
#             else:
#                 ness.run_individual_walks(matrix, seeds, uids, args.output, args.restart)


def _handle_seeds(options: types.Options) -> List[types.BioEntity]:
    """
    Handle and parse seed node user input.

    arguments:
        options: CLI options

    returns
        a list of parsed seed nodes, or an empty list if no seeds were provided
    """

    ## User provided individual seed nodes using the -s/--seed option
    if options.seeds:
        seeds = [types.BioEntity(s) for s in options.seeds]

    ## Parse the seed file
    elif options.seed_file:
        seeds = parse.read_seeds(options.seed_file)

    else:
        log._logger.warning('You did not provide any seed nodes...')
        log._logger.warning(
            'This will calculate all pairwise probabilities which could take awhile'
        )

        seeds = []

    return seeds


def _build_graph(options: types.Options) -> Tuple[Dict[types.BioEntity, int], csr_matrix]:
    """
    Parse inputs, build the heterogeneous graph, map bioentities to array positions,
    and convert the graph into a normalized adjacency matrix.

    arguments
        options: CLI options

    returns
        a tuple:
            0: a BioEntity -> unique ID mapping
            1: the heterogeneous graph in sparse matrix form
    """

    inputs = parse.read_inputs(options)
    uids, hetnet = graph.build_graph(inputs)
    matrix = graph.build_matrix(hetnet)

    return (uids, matrix)


def _handle_individual_walks(options, seeds, uids, matrix) -> None:
    """
    Handle any parallelization options and run individual random walks for each seed.

    arguments
        options: CLI options
        seeds:   seed nodes
        uids:    node -> uid mapping
        matrix:  graph adjacency matrix
    """

    ## Distribute the walks across cores
    if options.distributed:

        log._logger.info(f'Starting local cluster: {options.cores} cores...')

        ## Start a local cluster utilizing the specified number of cores
        client = Client(LocalCluster(
            n_workers=options.cores,
            processes=True,
            local_directory=tf.gettempdir()
        ))

        log._logger.info(f'Started local cluster...')

        ## Run the logging init function on each worker and register the callback so
        ## future workers also run the function
        init_logging_partial = partial(log._initialize_logging, verbose=options.verbose)
        client.register_worker_callbacks(setup=init_logging_partial)

        log._logger.info(f'Walking the graph...')

        ness.distribute_individual_walks(
            matrix,
            seeds,
            uids,
            options.output,
            alpha=options.restart,
            procs=options.cores
        )

        client.close()

    else:
        ness.run_individual_walks(matrix, seeds, uids, options.output, options.restart)


def _handle_permuted_individual_walks(options, seeds, uids, matrix) -> None:
    """
    Handle any parallelization options, run individual random walks for each seed, and
    run permutation testing for the results.

    arguments
        options: CLI options
        seeds:   seed nodes
        uids:    node -> uid mapping
        matrix:  graph adjacency matrix
    """

    ## Distribute permutation tests across a local cluster
    if options.distributed:

        ## Start a local cluster utilizing the specified number of cores
        client = Client(LocalCluster(
            n_workers=options.cores,
            processes=True,
            local_directory=tf.gettempdir()
        ))

        ## Run the logging init function on each worker and register the callback so
        ## future workers also run the function
        init_logging_partial = partial(log._initialize_logging, verbose=options.verbose)
        client.register_worker_callbacks(setup=init_logging_partial)

        ## Do the permutation testing
        ness.distribute_individual_permutation_tests(
            matrix,
            seeds,
            uids,
            options.output,
            permutations=options.permutations,
            alpha=options.restart,
            procs=options.cores,
        )

        client.close()

    else:
        ## Run the permutation testing
        ness.run_individual_permutation_tests(
            matrix, seeds, uids, options.output, options.permutations, options.restart
        )


def _handle_permuted_single_walk(options, seeds, uids, matrix) -> None:
    """
    Handle any parallelization options, run a single walk for all seeds, and run
    permutation testing for the results.

    arguments
        options: CLI options
        seeds:   seed nodes
        uids:    node -> uid mapping
        matrix:  graph adjacency matrix
    """

    ## Distribute permutation tests across a local cluster
    if options.distributed:

        ## Start a local cluster utilizing the specified number of cores
        client = Client(LocalCluster(
            n_workers=options.cores,
            processes=True,
            local_directory=tf.gettempdir()
        ))

        ## Run the logging init function on each worker and register the callback so
        ## future workers also run the function
        init_logging_partial = partial(log._initialize_logging, verbose=options.verbose)
        client.register_worker_callbacks(setup=init_logging_partial)

        ## Do the permutation testing
        ness.distribute_individual_permutation_tests(
            matrix,
            [seeds],
            uids,
            options.output,
            permutations=options.permutations,
            alpha=options.restart,
        )

        client.close()


@click.command()
@click.option(
    '-a',
    '--annotations',
    multiple=True,
    type=click.Path(readable=True, resolve_path=True),
    help='ontology annotation files'
)
@click.option(
    '-e',
    '--edges',
    multiple=True,
    type=click.Path(readable=True, resolve_path=True),
    help='edge list files'
)
@click.option(
    '-g',
    '--genesets',
    multiple=True,
    type=click.Path(readable=True, resolve_path=True),
    help='gene set files'
)
@click.option(
    '-h',
    '--homology',
    multiple=True,
    type=click.Path(readable=True, resolve_path=True),
    help='homology mapping files'
)
@click.option(
    '-o',
    '--ontologies',
    multiple=True,
    type=click.Path(readable=True, resolve_path=True),
    help='ontology relation files'
)
@click.option(
    '-s',
    '--seeds',
    multiple=True,
    type=str,
    help='seed node'
)
@click.option(
    '--seed-file',
    type=click.Path(exists=True, readable=True, resolve_path=True),
    help='file containing list of seed nodes'
)
@click.option(
    '-m',
    '--multiple',
    default=False,
    is_flag=True,
    help=(
        'if using multiple seed nodes, start from all seeds at once instead of '
        'one walk per seed'
    )
)
@click.option(
    '-d',
    '--distributed',
    default=False,
    is_flag=True,
    help='parallelize the random walk by distributing across available cores'
)
@click.option(
    '-c',
    '--cores',
    default=os.cpu_count(),
    help='distribute computations among N cores (default = all available cores)'
)
@click.option(
    '-n',
    '--permutations',
    default=0,
    type=int,
    help='permutations to run'
)
@click.option(
    '-r',
    '--restart',
    default=0.15,
    type=float,
    help='restart probability'
)
@click.option(
    '--verbose',
    default=False,
    is_flag=True,
    help='clutter your screen with output'
)
@click.version_option(version=__version__.__version__, prog_name='ness')
@click.argument('output', required=True)
def cli(
    annotations: Tuple[click.Path],
    edges: Tuple[click.Path],
    genesets: Tuple[click.Path],
    homology: Tuple[click.Path],
    ontologies: Tuple[click.Path],
    seeds: str,
    seed_file: click.Path,
    multiple: bool,
    distributed: bool,
    cores: int,
    permutations: int,
    restart: float,
    verbose: bool,
    output: click.Path
):
    """
    \b
    Network Enhanced Similarity Search (NESS).
    Integrate heterogeneous functional genomics datasets and calculate
    diffusion metrics over the heterogeneous network using a random
    walk with restart.
    """

    options = types.Options(
        annotations,
        edges,
        genesets,
        homology,
        ontologies,
        seeds,
        seed_file,
        multiple,
        distributed,
        cores,
        permutations,
        restart,
        verbose,
        output
    )

    log._initialize_logging(verbose=verbose)

    if not annotations and not edges and not genesets and not homology and not ontologies:
        log._logger.error('At least one of the following options must be specified:')
        log._logger.error('  --annotations, --edges, --genesets, --homlogy, --ontologies')
        exit(1)

    if seeds and seed_file:
        log._logger.error('Only one of --seeds, --seed-file can be used simultaneously')
        exit(1)

    seeds = _handle_seeds(options) # type: ignore
    uids, matrix = _build_graph(options)

    ## Use all nodes in the graph as the seeds
    if not seeds:
        seeds = list(uids.keys()) # type: ignore

    ## We're running one random walk per seed node and running permutation tests
    ## on the results of each RW
    if (not options.multiple) and options.permutations:
        _handle_permuted_individual_walks(options, seeds, uids, matrix)

    ## One random walk per seed, no permutation testing
    elif not options.multiple:
        _handle_individual_walks(options, seeds, uids, matrix)

    ## A single random walk starting from all seeds simultaneously with
    ## permutation testing
    elif options.multiple and options.permutations:
        pass

    elif options.multiple:
        pass

    else:
        pass


if __name__ == '__main__':
    # _main()
    cli()
