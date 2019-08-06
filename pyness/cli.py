#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: arguments.py
## desc: Argument parsing and handling.
## auth: TR

from dask.distributed import Client
from dask.distributed import LocalCluster
from functools import partial
import logging
import os

from . import arguments
from . import log
from . import graph
from . import ness
from . import parse

logging.getLogger(__name__).addHandler(logging.NullHandler())


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

    inputs = parse.read_inputs(args)
    uids, hetnet = graph.build_graph(inputs)
    matrix = graph.build_matrix(hetnet)

    ## Using all nodes in the graph
    if not seeds:
        seeds = uids.values()

    if args.multiple:
        if args.permutations:
            pass

        else:
            if args.distributed:
                log._logger.warning(
                    'Distributed computing is not need when starting from multiple seeds'
                )

    else:
        ## We're doing permutation testing...
        if args.permutations:

            ## Distribute permutation tests across a local cluster
            if args.distributed:

                ## Start a local cluster utilizing the specified number of cores
                client = Client(LocalCluster(n_workers=args.cores))

                ## Run the logging init function on each worker and register the callback so
                ## future workers also run the function
                init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
                client.register_worker_callbacks(setup=init_logging_partial)

                ## Do the permutation testing
                ness.distribute_individual_permutation_tests(
                    matrix, seeds, uids, args.output, args.permutations, args.restart
                )

            else:
                ## Run the permutation testing
                ness.run_individual_permutation_tests(
                    matrix, seeds, uids, args.output, args.permutations, args.restart
                )

        ## No permutation testing, just the walk...
        else:
            if args.distributed:
                ## Start a local cluster utilizing the specified number of cores
                client = Client(LocalCluster(n_workers=args.cores))

                ## Run the logging init function on each worker and register the callback so
                ## future workers also run the function
                init_logging_partial = partial(log._initialize_logging, verbose=args.verbose)
                client.register_worker_callbacks(setup=init_logging_partial)

                ness.distribute_individual_walks(
                    matrix, seeds, uids, args.output, args.restart
                )

            else:
                ness.run_individual_walks(matrix, seeds, uids, args.output, args.restart)


if __name__ == '__main__':
   _main()
