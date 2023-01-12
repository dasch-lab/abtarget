#!/usr/bin/env python

import os
import re
import argparse
import traceback
import sys
import numpy as np
import csv

from src import pdb as pdbParse

def read_files(path):
  
  res = []
  target = path.split('/')[-1].split('_')[2]
  for filename in os.listdir(path):
    pdb_path = os.path.join(path, filename)
    sequence_h = pdbParse.sequence(pdb_path, 'H', True)
    sequence_l = pdbParse.sequence(pdb_path, 'L', True)
    res.append([filename, sequence_h, sequence_l, target])
  
  return res

if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-i', '--input', help='input directory', dest='input', type=str, required=True)

  # Parse arguments
  #args = argparser.parse_args()
  #args.input = '/disk1/mAb_dataset/LH_Protein_Kabat'
  #input_path = '/disk1/mAb_dataset/LH_Protein_Kabat'

  print('Start')
  protein = read_files('/disk1/abtarget/mAb_dataset/NR_LH_Protein_Kabat')
  print('Protein: {}'.format(len(protein)))
  non_protein = read_files('/disk1/abtarget/mAb_dataset/NR_LH_NonProtein_Kabat')
  print('Non Protein: {}'.format(len(non_protein)))
  
  details = [['File_name', 'VH', 'VL', 'target']]

  print('Start writing')
  with open('/disk1/abtarget/mAb_dataset/dataset.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerows(details)
    mywriter.writerows(protein)
    print('Protein')
    mywriter.writerows(non_protein)
    print('Non-Protein')
    
