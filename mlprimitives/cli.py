# -*- coding: utf-8 -*-

"""MLPrimitives Command Line Interface module."""

import argparse
import logging
import os
import sys
import warnings

from mlblocks import add_primitives_path, get_primitives_paths

from mlprimitives.evaluation import score_pipeline

LOGGER = logging.getLogger(__name__)


def _logging_setup(verbosity=1):
    logger = logging.getLogger()
    log_level = (4 - verbosity) * 10
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _test(args):
    for pipeline in args.pipeline:
        print('Scoring pipeline: {}'.format(pipeline))
        score, stdev = score_pipeline(pipeline, args.splits)
        print('Obtained Score: {:.4f} +/- {:.4f}'.format(score, stdev))


def _get_primitives(pattern):
    primitives = list()
    for base_path in get_primitives_paths():
        if os.path.exists(base_path):
            for filename in os.listdir(base_path):
                if pattern in filename and filename.endswith('.json'):
                    primitives.append(filename[:-5])

    return list(sorted(primitives))


def _list(args):
    print('\n'.join(_get_primitives(args.pattern)))


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        sys.stderr.write('\nERROR: {}\n\n'.format(message))
        self.print_help()
        sys.exit(2)


def _get_parser():
    parser = ArgumentParser(
        description='MLPrimitives Command Line Interface')

    parser.add_argument(
        '-p', '--primitives-path', action='append', help=(
            'Path where primitives should be looked for. Use multiple '
            'times in order to add multiple directories'
        )
    )
    parser.add_argument('-v', '--verbose', action='count', default=0)

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    subparser = subparsers.add_parser('test', help='Test a single pipeline.')
    subparser.set_defaults(action=_test)
    subparser.add_argument('-s', '--splits', default=1, type=int,
                           help='Number of splits to use for Cross Validation')
    subparser.add_argument('pipeline', nargs='+')

    subparser = subparsers.add_parser('list', help='List available primitives')
    subparser.set_defaults(action=_list)
    subparser.add_argument('pattern', nargs='?', default='')

    return parser


def _add_primitives_paths(paths):
    if paths:
        for path in paths:
            add_primitives_path(path)


def _process_common_args(args):
    _add_primitives_paths(args.primitives_path)
    _logging_setup(args.verbose)


def main():
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    parser = _get_parser()
    args = parser.parse_args()
    if not args.action:
        parser.print_help()
        sys.exit(0)

    _process_common_args(args)

    args.action(args)
