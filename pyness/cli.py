#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: arguments.py
## desc: Argument parsing and handling.
## auth: TR

from dask.distributed import Client
from dask.distributed import LocalCluster
from dask.distributed import get_client
from functools import partial
from time import sleep
import logging
import tempfile as tf

from . import arguments
from . import log
from . import graph
from . import ness
from . import parse

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _wait_for_workers(n, min_n=0, limit=30, sleepy=5):

    client = get_client()
    waits = 0
    num_started = 0

    while True:
        if limit > 0 and waits > limit:
            log._logger.warning(
                'Waiting time limit has been reached, starting without additional workers'
            )
            break

        workers =  client.scheduler_info()["workers"].keys()

        sleep(sleepy)

        if len(workers) > 0 and len(workers) != num_started:
            num_started = len(workers)

            log._logger.info(f'Started {len(workers)} workers...')

        if len(workers) >= n:
            log._logger.info(f'All {len(workers)} workers have started...')
            break

        if min_n > 0 and len(workers) >= min_n:
            break

        waits += 1


def _main() -> None:
    """
    Application main.
    """

    args = arguments.setup_arguments('pyness')

    log._initialize_logging(verbose=args.verbose)

    if args.seeds:
        seeds = args.seeds

    elif args.seed_file:
        seeds = parse.read_seeds(args.seed_file)

    else:
        log._logger.warning('You did not provide any seed nodes...')
        log._logger.warning(
            'This will calculate all pairwise probabilities which could take awhile'
        )

        seeds = []

    #print(seeds[:5])
    inputs = parse.read_inputs(args)
    uids, hetnet = graph.build_graph(inputs)
    matrix = graph.build_matrix(hetnet)
    #print(list(uids.keys())[:5])

    #for s in seeds:
    #    if s not in uids:
    #        print(f'{s} not in uid')

    #for s in uids.keys():
    #    if s.biotype == 'geneset':
    #        print(s)

    ## Using all nodes in the graph
    if not seeds:
        seeds = list(uids.keys())

    if args.multiple:
        if args.permutations:

            ## Start a local cluster utilizing the specified number of cores
            client = Client(LocalCluster(
                n_workers=args.cores,
                processes=True,
                local_directory=tf.gettempdir()
            ))

            ## Run the logging init function on each worker and register the callback so
            ## future workers also run the function
            init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
            client.register_worker_callbacks(setup=init_logging_partial)

            ## Do the permutation testing
            ness.distribute_individual_permutation_tests(
                matrix,
                [seeds],
                uids,
                args.output,
                permutations=args.permutations,
                alpha=args.restart,
                fdr=args.fdr
            )

            client.close()

        else:
            if args.distributed:
                log._logger.warning((
                    'Distributed computing is unnecessary when starting from '
                    'multiple seeds'
                ))

    else:
        ## We're doing permutation testing...
        if args.permutations:

            ## Distribute permutation tests across a local cluster
            if args.distributed:

                ## Start a local cluster utilizing the specified number of cores
                client = Client(LocalCluster(
                    n_workers=args.cores,
                    processes=True,
                    local_directory=tf.gettempdir()
                ))

                ## Run the logging init function on each worker and register the callback so
                ## future workers also run the function
                init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
                client.register_worker_callbacks(setup=init_logging_partial)

                ## Do the permutation testing
                ness.distribute_individual_permutation_tests(
                    matrix,
                    seeds,
                    uids,
                    args.output,
                    permutations=args.permutations,
                    alpha=args.restart,
                    procs=args.cores,
                    fdr=args.fdr
                )

                client.close()

            else:
                ## Run the permutation testing
                ness.run_individual_permutation_tests(
                    matrix, seeds, uids, args.output, args.permutations, args.restart
                )

        ## No permutation testing, just the walk...
        else:
            if args.distributed:

                log._logger.info(f'Starting local cluster: {args.cores} cores...')

                ## Start a local cluster utilizing the specified number of cores
                client = Client(LocalCluster(
                    n_workers=args.cores,
                    processes=True,
                    local_directory=tf.gettempdir()
                ))

                log._logger.info(f'Started local cluster...')

                _wait_for_workers(args.cores, limit=0, sleepy=2)

                ## Run the logging init function on each worker and register the callback so
                ## future workers also run the function
                init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
                client.register_worker_callbacks(setup=init_logging_partial)

                log._logger.info(f'Walking the graph...')

                ness.distribute_individual_walks(
                    matrix, seeds, uids, args.output, alpha=args.restart, procs=args.cores
                )

                client.close()

            else:
                ness.run_individual_walks(matrix, seeds, uids, args.output, args.restart)


if __name__ == '__main__':
   _main()
