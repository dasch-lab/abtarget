#!/usr/bin/env python

import os
import re
import argparse
import traceback
import sys
import numpy as np
import csv
import pandas as pd

from src.pdb import sequence
from random import sample
import urllib.request
import urllib.parse
from antiberty import AntiBERTyRunner
from src.protbert import BaselineOne
import torch
from matplotlib import pyplot as plt
import umap
import seaborn as sns


def read_files(path, n_class, chainH, chainL):
  
  res = []
  target = path.split('/')[-1]
  for filename in os.listdir(path):
    pdb_path = os.path.join(path, filename)
    sequence_h = sequence(pdb_path, chainH, False)
    sequence_l = sequence(pdb_path, chainL, False)
    res.append([filename, sequence_h, sequence_l, target, n_class])
  
  return res

def __getPDB(pdb, output_path):
 
  try:
 
    output_file = os.path.join(output_path, '{}.pdb'.format(pdb.lower()))
    if not os.path.exists(output_file):
 
      URL = "https://files.rcsb.org/view/{}.pdb".format(pdb.upper())
 
      # Download file
      response = urllib.request.urlopen(URL)
      with open(output_file, 'wb') as handle:
        handle.write(response.read())
     
      return output_file
   
  except Exception as e:
    traceback.print_exc()
    return None

def csv_union(path1, path2, save_path):
  files = [path1, path2]
  df = pd.DataFrame()
  for file in files:
    data = pd.read_csv(file)
    df = pd.concat([df, data], axis=0)
  df.to_csv(save_path, header = True, index=False)

def pdb2seq(pdb_names, save_path, df):
  for name in pdb_names:
    if name not in df['pdb']:

      num = len(df[df['pdb'] == name])
      for ind in range(num):
        data = df[df['pdb'] == name].iloc[ind:ind+1]

        if data['antigen_type'].to_string(index=False) in protein_list:
        #pdb_file = __getPDB(data['pdb'].to_string(index=False), os.path.join(output_path,'protein'))
          pdb_path =  os.path.join(output_path, 'protein' , name + '.pdb')
          try:
            sequence_h = sequence(pdb_path, data['Hchain'].to_string(index=False), False)
            sequence_l = sequence(pdb_path, data['Lchain'].to_string(index=False), False)
            if len(sequence_h) == 0 or len(sequence_l) == 0:
              pass
            else:
              res.append([name, sequence_h, sequence_l, data['antigen_type'].to_string(index=False), 0])
              break 
          except OSError as e:
            pass
        elif data['antigen_type'].to_string(index=False) in nonprotein_list:
          pdb_path =  os.path.join(output_path, 'nonprotein' , name + '.pdb')
          try:
            sequence_h = sequence(pdb_path, data['Hchain'].to_string(index=False), False)
            sequence_l = sequence(pdb_path, data['Lchain'].to_string(index=False), False)
            if len(sequence_h) == 0 or len(sequence_l) == 0:
              pass
            else:
              res.append([name, sequence_h, sequence_l, data['antigen_type'].to_string(index=False), 1])
              break
          except OSError as e:
            pass
  
  df = pd.DataFrame(res, columns = ['name', 'VH', 'VL', 'target', 'label'])
  df.to_csv(save_path, header=False, index = False)

def xlsx2csv(in_path, sheet_name, columns_names, out_path):
  dfs = pd.read_excel(in_path, sheet_name=sheet_name)
  name = set(dfs['sample'])
  df_fin = pd.DataFrame(columns = columns_names)
  #df_list = []

  for inx, el in enumerate(name):
    df = dfs[dfs['sample'] == el]
    vl = df[df['type'] == 'light']['protein'].values[0]
    vh = df[df['type'] == 'heavy']['protein'].values[0]
    df_line = pd.DataFrame({'name': el, 'VH': vh, 'VL': vl, 'target': 'NonProtein', 'label': 1}, index = [inx])
    df_fin = pd.concat([df_fin, df_line])
    #df_fin = pd.concat([df_fin, df_line])
    #df_fin.append({'name': el, 'VH': vh, 'VL': vl, 'target': 'NonProtein', 'label': 1}, ignore_index=True)
  
  df_fin.to_csv(out_path, index=False)
  return 

'''def read_txt(in_file; out_file):
  df = pd.read_csv(in_file, sep='\t',usecols=[0,1,2,5], index_col=False)
  df_new = df.dropna()
  print(len(df_new))
  df_new = df_new[df_new['antigen_type'] != 'unknown']
  print(len(df_new))
  df_new.to_csv(out_file, index=False)'''


def sample_df(df_protein, df_nonprotein, num, random):
  df_protein = df_protein.sample(n=num, random_state=random)
  print(len(df_protein))
  df_nonprotein = df_nonprotein.sample(n=num, random_state=random)
  print(len(df_nonprotein))
  return df_protein, df_nonprotein

def filter_df(df_protein, df_nonprotein, df_protein_sample, df_nonprotein_sample):
  df_protein = df_protein[~df_protein['name'].isin(df_protein_sample['name'])]
  print(len(df_protein))
  df_nonprotein = df_nonprotein[~df_nonprotein['name'].isin(df_nonprotein_sample['name'])]
  print(len(df_nonprotein))

  return df_protein, df_nonprotein

def df2csv(df_protein, df_nonprotein, out_file):
  df_protein.to_csv(out_file, header=['name', 'VH', 'VL', 'target', 'label'], index=False)
  df_nonprotein.to_csv(out_file, mode='a', header=False, index = False)


def split_dataset(input_path, perc_val, perc_test, output_path):
  df = pd.read_csv(input_path)
  df_protein = df[df['label'] == 0]
  print(len(df_protein))
  df_nonprotein = df[df['label'] == 1]
  print(len(df_nonprotein))

  print('Test')
  df_protein_test, df_nonprotein_test = sample_df(df_protein, df_nonprotein, round(len(df_nonprotein)*perc_test), 42)

  df_protein_train_val, df_nonprotein_train_val = filter_df(df_protein, df_nonprotein, df_protein_test, df_nonprotein_test)

  print('Val')
  df_protein_val, df_nonprotein_val = sample_df(df_protein_train_val, df_nonprotein_train_val, round(len(df_nonprotein)*perc_val), 42)

  print('Training')
  df_protein_train, df_nonprotein_train = filter_df(df_protein_train_val, df_nonprotein_train_val, df_protein_val, df_nonprotein_val)

  perc_train = 1 - perc_val - perc_test
  df2csv(df_protein_test, df_nonprotein_test, output_path + '/sabdab_200423_test_'+ str(perc_test) + '.csv')
  df2csv(df_protein_val, df_nonprotein_val, output_path + '/sabdab_200423_val_'+ str(perc_val) + '.csv')
  df2csv(df_protein_train, df_nonprotein_train, output_path + '/sabdab_200423_train_'+ str(perc_train) + '.csv')



if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-i', '--input', help='input directory', dest='input', type=str, required=True)

  print('Start')

  #data = pd.read_csv('/disk1/abtarget/dataset/sabdab_200423.txt', sep='/t')

  output_path = '/disk1/abtarget/dataset/sabdab'

  protein_list = ['peptide | peptide | peptide', 'peptide | protein | protein', 'peptide | protein', 'protein', 'protein | peptide', 'protein | protein | protein | protein', 
                  'protein | peptide | protein', 'protein | protein', 'protein | protein | protein', 'peptide | peptide', 'peptide', 'protein | protein | protein | peptide',
                  'protein | protein | protein | protein | protein', 'protein | protein | peptide']
  
  hybrid_list = ['carbohydrate | protein', 'carbohydrate | protein | protein', 'protein | nucleic-acid']

  nonprotein_list = ['Hapten', 'carbohydrate', 'nucleic-acid', 'nucleic-acid | nucleic-acid | nucleic-acid', 'nucleic-acid | nucleic-acid']
  res = []
  non_seq = []

  
  #list_nonprotein = ['1h8s', '3gkz', '3gm0', '4gqp', '4h0i', '4lar', '4las', '5i4f', '5j74', '5j75', '6k4z', '6lfw', '6lfx', '6uy3', '7f35', '7rdm']

  #split_dataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep.csv', 0.1, 0.1, '/disk1/abtarget/dataset/sabdab/split')

  df_dataset = pd.read_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep.csv')

  '''tot_target = protein_list
  tot_target.extend(nonprotein_list)
  df_new_dataset = pd.DataFrame()

  device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

  model = BaselineOne(1, device, nn_classes=2, freeze_bert=True, model_name='protbert')

  for ind, type in enumerate(tot_target):
    df_type = df_dataset[df_dataset['target'] == type]
    df_type['type_label'] = ind 
    df_new_dataset.append(df_type, ignore_index = True)
  
  labels = list(df_new_dataset['type_label'])

  model.eval()
  embeddings = []

  for index in range(len(df_new_dataset)):
    embeddings.append(np.squeeze(model(df_new_dataset.loc[index]).cpu().detach().numpy()))
  
  reducer = umap.UMAP()
  embedding = reducer.fit_transform(embeddings)
  embedding.shape

  plt.scatter(embedding[:, 0], embedding[:, 1],c=[sns.color_palette()[x] for x in labels])
  plt.gca().set_aspect('equal', 'datalim')
  plt.title('UMAP projection', fontsize=24)
  plt.savefig('umap.jpg')'''

  '''l = [['6dc8', 'peptide', 0, 1, 'AntiMLP'], ['4rgo', 'protein', 0, 1, 'AntiMLP'], ['6a0z', 'protein', 0, 1, 'AntiMLP'], ['6i3z', 'protein | peptide', 0, 1, 'AntiMLP'], ['5esz', 'protein', 0, 1, 'AntiMLP'], 
       ['6pxr', 'peptide', 0, 1, 'AntiMLP'], ['8dcy', 'protein', 0, 1, 'AntiMLP'], ['7vng', 'protein', 0, 1, 'AntiMLP'], ['3kdm', 'Hapten', 1, 0, 'AntiMLP'], ['43ca', 'Hapten', 1, 0, 'AntiMLP'], 
       ['4hih', 'carbohydrate', 1, 0, 'AntiMLP'], ['2jb6', 'Hapten', 1, 0, 'AntiMLP'], ['6zec', 'carbohydrate', 1, 0, 'AntiMLP'], ['4jn2', 'Hapten', 1, 0, 'AntiMLP'], ['2a9n', 'Hapten', 1, 0, 'AntiMLP'], 
       ['1uwg', 'Hapten', 1, 0, 'AntiMLP'], ['2jb5', 'Hapten', 1, 0, 'AntiMLP']]
  
  cross1 = [['6dc8', 'peptide', 0, 1, 'cross1'], ['6pxr', 'peptide', 0, 1, 'cross1'], ['2fbj', 'carbohydrate', 1, 0, 'cross1'], ['6nfn', 'Hapten', 1, 0, 'cross1'], ['5nbw', 'Hapten', 1, 0, 'cross1'], 
            ['3kdm', 'Hapten', 1, 0, 'cross1'], ['43ca', 'Hapten', 1, 0, 'cross1'], ['3mug', 'carbohydrate', 1, 0, 'cross1'], ['4fz8', 'carbohydrate', 1, 0, 'cross1'], ['5dqd', 'carbohydrate', 1, 0, 'cross1'], 
            ['5cp3', 'Hapten', 1, 0, 'cross1'], ['4hih', 'carbohydrate', 1, 0, 'cross1'], ['2jb6', 'Hapten', 1, 0, 'cross1'], ['6zec', 'carbohydrate', 1, 0, 'cross1'], ['4jn2', 'Hapten', 1, 0, 'cross1'], 
            ['5bzw', 'carbohydrate', 1, 0, 'cross1'], ['2a9n', 'Hapten', 1, 0, 'cross1'], ['7u63', 'Hapten', 1, 0, 'cross1'], ['3cfd', 'Hapten', 1, 0, 'cross1'], ['2ykl', 'Hapten', 1, 0, 'cross1'], 
            ['1ngp', 'Hapten', 1, 0, 'cross1'], ['1uwg', 'Hapten', 1, 0, 'cross1'], ['2ajv', 'Hapten', 1, 0, 'cross1'], ['2jb5', 'Hapten', 1, 0, 'cross1']]
  
  cross2 = [['6dc8', 'peptide', 0, 1, 'cross2'], ['6pxr', 'peptide', 0, 1, 'cross2'], ['2fbj', 'carbohydrate', 1, 0, 'cross2'], ['6nfn', 'Hapten', 1, 0, 'cross2'], ['5nbw', 'Hapten', 1, 0, 'cross2'],
            ['3kdm', 'Hapten', 1, 0, 'cross2'], ['43ca', 'Hapten', 1, 0, 'cross2'], ['6uul', 'carbohydrate', 1, 0, 'cross2'], ['4fz8', 'carbohydrate', 1, 0, 'cross2'], ['5cp3', 'Hapten', 1, 0, 'cross2'], 
            ['4hih', 'carbohydrate', 1, 0, 'cross2'], ['2jb6', 'Hapten', 1, 0, 'cross2'], ['6zec', 'carbohydrate', 1, 0, 'cross2'], ['4jn2', 'Hapten', 1, 0, 'cross2'], ['5bzw', 'carbohydrate', 1, 0, 'cross2'], 
            ['2a9n', 'Hapten', 1, 0, 'cross2'], ['5xqw', 'Hapten', 1, 0, 'cross2'], ['1fl3', 'Hapten', 1, 0, 'cross2'], ['2mcp', 'Hapten', 1, 0, 'cross2'], ['7u63', 'Hapten', 1, 0, 'cross2'], 
            ['3cfd', 'Hapten', 1, 0, 'cross2'], ['2ykl', 'Hapten', 1, 0, 'cross2'], ['1uwg', 'Hapten', 1, 0, 'cross2'], ['2ajv', 'Hapten', 1, 0, 'cross2'], ['2jb5', 'Hapten', 1, 0, 'cross2']]
  
  cross3 = [['6dc8', 'peptide', 0, 1, 'cross3'], ['6pxr', 'peptide', 0, 1, 'cross3'], ['2fbj', 'carbohydrate', 1, 0, 'cross3'], ['6nfn', 'Hapten', 1, 0, 'cross3'], ['5nbw', 'Hapten', 1, 0, 'cross3'], 
            ['3kdm', 'Hapten', 1, 0, 'cross3'], ['4uin', 'Hapten', 1, 0, 'cross3'], ['43ca', 'Hapten', 1, 0, 'cross3'], ['6uul', 'carbohydrate', 1, 0, 'cross3'], ['4fz8', 'carbohydrate', 1, 0, 'cross3'], 
            ['5cp3', 'Hapten', 1, 0, 'cross3'], ['4hih', 'carbohydrate', 1, 0, 'cross3'], ['2jb6', 'Hapten', 1, 0, 'cross3'], ['6zec', 'carbohydrate', 1, 0, 'cross3'], ['4jn2', 'Hapten', 1, 0, 'cross3'], 
            ['5bzw', 'carbohydrate', 1, 0, 'cross3'], ['2a9n', 'Hapten', 1, 0, 'cross3'], ['1fl3', 'Hapten', 1, 0, 'cross3'], ['2mcp', 'Hapten', 1, 0, 'cross3'], ['7u63', 'Hapten', 1, 0, 'cross3'], 
            ['3cfd', 'Hapten', 1, 0, 'cross3'], ['1ngp', 'Hapten', 1, 0, 'cross3'], ['1uwg', 'Hapten', 1, 0, 'cross3'], ['2ajv', 'Hapten', 1, 0, 'cross3'], ['2jb5', 'Hapten', 1, 0, 'cross3']]
  
  cross4 = [['6dc8', 'peptide', 0, 1, 'cross4'], ['6pxr', 'peptide', 0, 1, 'cross4'], ['2fbj', 'carbohydrate', 1, 0, 'cross4'], ['6nfn', 'Hapten', 1, 0, 'cross4'], ['5nbw', 'Hapten', 1, 0, 'cross4'], 
            ['3kdm', 'Hapten', 1, 0, 'cross4'], ['4uin', 'Hapten', 1, 0, 'cross4'], ['43ca', 'Hapten', 1, 0, 'cross4'], ['6uul', 'carbohydrate', 1, 0, 'cross4'], ['4fz8', 'carbohydrate', 1, 0, 'cross4'], 
            ['5dqd', 'carbohydrate', 1, 0, 'cross4'], ['5cp3', 'Hapten', 1, 0, 'cross4'], ['4hih', 'carbohydrate', 1, 0, 'cross4'], ['2jb6', 'Hapten', 1, 0, 'cross4'], ['6zec', 'carbohydrate', 1, 0, 'cross4'], 
            ['4jn2', 'Hapten', 1, 0, 'cross4'], ['5bzw', 'carbohydrate', 1, 0, 'cross4'], ['2a9n', 'Hapten', 1, 0, 'cross4'], ['5xqw', 'Hapten', 1, 0, 'cross4'], ['1fl3', 'Hapten', 1, 0, 'cross4'], 
            ['2mcp', 'Hapten', 1, 0, 'cross4'], ['7u63', 'Hapten', 1, 0, 'cross4'], ['3cfd', 'Hapten', 1, 0, 'cross4'], ['1ngp', 'Hapten', 1, 0, 'cross4'], ['1uwg', 'Hapten', 1, 0, 'cross4'], 
            ['2ajv', 'Hapten', 1, 0, 'cross4'], ['2jb5', 'Hapten', 1, 0, 'cross4']]
  
  cross5 = [['6dc8', 'peptide', 0, 1, 'cross5'], ['6pxr', 'peptide', 0, 1, 'cross5'], ['2fbj', 'carbohydrate', 1, 0, 'cross5'], ['6nfn', 'Hapten', 1, 0, 'cross5'], ['5nbw', 'Hapten', 1, 0, 'cross5'], 
            ['3kdm', 'Hapten', 1, 0, 'cross5'], ['43ca', 'Hapten', 1, 0, 'cross5'], ['6uul', 'carbohydrate', 1, 0, 'cross5'], ['4fz8', 'carbohydrate', 1, 0, 'cross5'], ['5cp3', 'Hapten', 1, 0, 'cross5'], 
            ['4hih', 'carbohydrate', 1, 0, 'cross5'], ['2jb6', 'Hapten', 1, 0, 'cross5'], ['6zec', 'carbohydrate', 1, 0, 'cross5'], ['4jn2', 'Hapten', 1, 0, 'cross5'], ['5bzw', 'carbohydrate', 1, 0, 'cross5'], 
            ['2a9n', 'Hapten', 1, 0, 'cross5'], ['5xqw', 'Hapten', 1, 0, 'cross5'], ['1fl3', 'Hapten', 1, 0, 'cross5'], ['2mcp', 'Hapten', 1, 0, 'cross5'], ['7u63', 'Hapten', 1, 0, 'cross5'], 
            ['3cfd', 'Hapten', 1, 0, 'cross5'], ['1ngp', 'Hapten', 1, 0, 'cross5'], ['1uwg', 'Hapten', 1, 0, 'cross5'], ['2ajv', 'Hapten', 1, 0, 'cross5'], ['2jb5', 'Hapten', 1, 0, 'cross5']]
  


  df_l = pd.DataFrame(l, columns = ['pdb', 'target', 'true', 'predicted', 'model'])
  df_cross1 = pd.DataFrame(cross1, columns = ['pdb', 'target', 'true', 'predicted', 'model'])
  df_cross2 = pd.DataFrame(cross2, columns = ['pdb', 'target', 'true', 'predicted', 'model'])
  df_cross3 = pd.DataFrame(cross3, columns = ['pdb', 'target', 'true', 'predicted', 'model'])
  df_cross4 = pd.DataFrame(cross4, columns = ['pdb', 'target', 'true', 'predicted', 'model'])
  df_cross5 = pd.DataFrame(cross5, columns = ['pdb', 'target', 'true', 'predicted', 'model'])

  df_tot = df_l


  df_tot = df_tot.append(df_cross1, ignore_index = True)
  df_tot = df_tot.append(df_cross2, ignore_index = True)
  df_tot = df_tot.append(df_cross3, ignore_index = True)
  df_tot = df_tot.append(df_cross4, ignore_index = True)
  df_tot = df_tot.append(df_cross5, ignore_index = True)

  list_pdb = list(set(df_tot['pdb']))

  #df_ordered = df_tot.groupby(by = ['pdb', 'target', 'true', 'predicted'])

  #df_tot = df_tot.join(df_tot['pdb'].apply(lambda l: ~pd.Series({i:i for i in l}).isna()).fillna(False).astype(int))
  df_tot = df_tot.groupby(['pdb', 'target','model']).size().unstack(fill_value=0).astype(int).rename_axis(None, axis=1).reset_index()
  df_tot.to_excel("output.xlsx")'''

  df_test = pd.read_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_val_norep.csv', sep=',')
  df_prtoein_test = df_test[df_test['label'] == 0]
  df_nonprtoein_test = df_test[df_test['label'] == 1]
  df_training = pd.read_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep.csv', sep=',')
  df_prtoein_train = df_training[df_training['label'] == 0]
  df_nonprtoein_train = df_training[df_training['label'] == 1]
  filter_df(df_prtoein_train, df_nonprtoein_train, df_prtoein_test, df_nonprtoein_test)





  print('Done')

  
    
