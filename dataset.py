#!/usr/bin/env python

import os
import re
import argparse
import traceback
import sys

from src import pdb as pdbParse

if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-i', '--input', help='input directory', dest='input', type=str, required=True)

  # Parse arguments
  args = argparser.parse_args()

  for filename in os.listdir(args.input):
    # Check file type
    # Parse PDB file
    # Store sequence data  

    # Extract data from RCSB
    for chain in ['H', 'L']:
      sequence = pdbParse.sequence(pdb_path, chain, True)

    pass