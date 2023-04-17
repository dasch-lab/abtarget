#!/usr/bin/env python

import os
import re
import argparse
import traceback
import sys
import numpy as np
import csv
import pandas as pd

from src import pdb as pdbParse
from random import sample

def read_files(path, n_class):
  
  res = []
  target = path.split('/')[-1].split('_')[2]
  for filename in os.listdir(path):
    pdb_path = os.path.join(path, filename)
    sequence_h = pdbParse.sequence(pdb_path, 'H', False)
    sequence_l = pdbParse.sequence(pdb_path, 'L', False)
    res.append([filename, sequence_h, sequence_l, target, n_class])
  
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
  #protein = read_files('/disk1/abtarget/dataset/NR_LH_Protein_Kabat', 0)
  #print('Protein: {}'.format(len(protein)))
  #non_protein = read_files('/disk1/abtarget/dataset/NR_LH_NonProtein_Kabat', 1)
  #print('Non Protein: {}'.format(len(non_protein)))


  '''xls = pd.ExcelFile('/disk1/abtarget/dataset/abtarget_dataset.xlsx')
  df1 = pd.read_excel(xls, 'patent_US10000567B2')
  df1["name_id"] = df1['id'].astype(str) +"_"+ df1["name"].astype(str)
  df2 = pd.read_excel(xls, 'cov-abdab')
  df2["name_id"] = df2['id'].astype(str) +"_"+ df2["name"].astype(str)

  df1_new=df1[['name_id','VH','VL','target']].rename({'name_id':'name','VH':'VH','VL':'VL','target':'target'}, axis=1)
  df2_new=df2[['name_id','VH','VL','target']].rename({'name_id':'name','VH':'VH','VL':'VL','target':'target'}, axis=1)

  df_new = pd.concat([df1_new, df2_new])
  df_new['label'] = 0
  
  details = [['name', 'VH', 'VL', 'target', 'label']]

  df_new.to_csv('/disk1/abtarget/dataset/split/train_aug_gm.csv', mode='a', index=False, header=False)'''

  '''print('Start writing')
  with open('/disk1/abtarget/dataset/abdb_dataset_protein_gm.csv', 'w') as f:
    mywriter = csv.writer(f, delimiter=',')
    mywriter.writerows(details)
    mywriter.writerows(protein)
    print('Protein')
    mywriter.writerows(non_protein)
    print('Non-Protein')'''
  
  df = pd.read_csv('/disk1/abtarget/dataset/abdb_dataset_old.csv')
  df_prot = df[df['target']=='Protein']
  df_noprot = df[df['target']=='NonProtein']
  
  df_prot['label'] = 0
  df_noprot['label'] = 1

  trainLen = round(len(df_noprot.index)*0.8)+2
  testLen = round(len(df_noprot.index)*0.1)-1

  idx = sample(range(len(df_noprot.index)), len(df_noprot.index))
  df_noprot_train = df_noprot.iloc[idx[:trainLen],:]
  df_noprot_val = df_noprot.iloc[idx[trainLen:trainLen+testLen],:]
  df_noprot_test = df_noprot.iloc[idx[trainLen+testLen:]]

  idx = sample(range(len(df_prot.index)), len(df_prot.index))
  df_prot_train = df_prot.iloc[idx[:-2*testLen]]
  df_prot_val = df_prot.iloc[idx[-2*testLen: -testLen]]
  df_prot_test = df_prot.iloc[idx[len(df_prot_train.index)+len(df_prot_val.index):]]

  df_prot_train.to_csv('/disk1/abtarget/dataset/new_split/train.csv', index=False)
  df_noprot_train.to_csv('/disk1/abtarget/dataset/new_split/train.csv', mode='a', index=False, header=False)

  df_prot_val.to_csv('/disk1/abtarget/dataset/new_split/validation.csv', index=False)
  df_noprot_val.to_csv('/disk1/abtarget/dataset/new_split/validation.csv', mode='a', index=False, header=False)

  df_prot_test.to_csv('/disk1/abtarget/dataset/new_split/test.csv', index=False)
  df_noprot_test.to_csv('/disk1/abtarget/dataset/new_split/test.csv', mode='a', index=False, header=False)


  print('Done')

  
    
