#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## file: arguments.py
## desc: Argument parsing and handling.
## auth: TR

from argparse import ArgumentParser
from typing import List
import logging
import os

from . import parse

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _parse_comma_list(cl: str) -> List[str]:
    """
    Parses a comma delimited string into an actual list.

    arguments
        cl: comma separated string

    returns
        a list
    """

    return [s.strip() for s in cl.split(',')]


def _register_args(parser: ArgumentParser) -> ArgumentParser:
    """

    :param parser:
    :return:
    """

    ## Options common to all (sub)parsers
    parser = ArgumentParser(add_help=False)
    input_group = parser.add_argument_group('input options')
    proc_group = parser.add_argument_group('processing options')

    parser.add_argument(
        'output',
        nargs='?',
        help='output filepath'
    )

    input_group.add_argument(
        '-a',
        '--annotations',
        action='append',
        dest='annotations',
        help='annotation files'
    )

    input_group.add_argument(
        '-e',
        '--edges',
        action='append',
        dest='edges',
        help='edge list files'
    )

    input_group.add_argument(
        '-g',
        '--genesets',
        action='append',
        dest='genesets',
        help='gene set files'
    )

    input_group.add_argument(
        '-h',
        '--homology',
        action='append',
        dest='homology',
        help='homology files'
    )

    input_group.add_argument(
        '-o',
        '--ontologies',
        action='append',
        dest='ontologies',
        help='ontology relationship files'
    )

    input_group.add_argument(
        '-s',
        '--seeds',
        action='store',
        dest='seeds',
        type=str,
        help='comma delimited list of seed nodes'
    )

    input_group.add_argument(
        '--seed-file',
        action='store',
        dest='seed_file',
        type=str,
        help='file containing seed nodes'
    )

    proc_group.add_argument(
        '-m',
        '--multiple',
        action='store_true',
        dest='multiple',
        help=(
            'if using multiple seed nodes, start from all seeds at once instead of '
            'one walk per seed'
        )
    )

    proc_group.add_argument(
        '-d',
        '--distributed',
        action='store_true',
        dest='distributed',
        help='parallelize the random walk using all available cores'
    )

    proc_group.add_argument(
        '-c',
        '--cores',
        action='store',
        default=os.cpu_count(),
        dest='cores',
        type=int,
        help='distribute computations among N cores (default = all available cores)'
    )

    proc_group.add_argument(
        '-n',
        '--permutations',
        action='store',
        default=0,
        dest='permutations',
        type=int,
        help='permutations to run'
    )

    proc_group.add_argument(
        '-r',
        '--restart',
        action='store',
        default=0.15,
        dest='restart',
        type=float,
        help='restart probability'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',
        help='clutter your screen with output'
    )

    return parser


def _validate_arguments(parser: ArgumentParser) -> None:
    """
    Ensures arguemnts passed to the CLI script are valid. Informs the user of any missing
    or invalid arguments.

    :param args:
    :return:
    """

    args = parser.parse_args()

    if not args.annotations and\
       not args.edges and\
       not args.genesets and\
       not args.homology and\
       not args.ontologies:
           print('')
           print('ERROR: You need to provide at least one input type')
           print('')
           parser.print_help()
           exit(1)

    if not args.output:
        print('')
        print('ERROR: You need to provide an output filepath')
        print('')
        parser.print_help()
        exit(1)

    return args


def setup_arguments(exe):
    """

    :param exe:
    :return:
    """

    parser = ArgumentParser()
    parser = _register_args(parser)

    args = _validate_arguments(parser)

    if args.seeds is not None:
        args.seeds = _parse_comma_list(args.seeds)
        args.seeds = [parse.parse_seed(s) for s in args.seeds]

    return args

