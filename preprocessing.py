#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Foobar.py: Description of what foobar does."""

__author__      = "Mirek Buddha"
__copyright__   = "Copyright 2022, Data Mind s.r.o."

import os
import sys
import argparse


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)

    print(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))