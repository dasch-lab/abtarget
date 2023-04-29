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
  
  '''df = pd.read_csv('/disk1/abtarget/dataset/abdb_dataset_old.csv')
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
  df_noprot_test.to_csv('/disk1/abtarget/dataset/new_split/test.csv', mode='a', index=False, header=False)'''



  '''files = ['/disk1/abtarget/dataset/new_split/train.csv', '/disk1/abtarget/dataset/new_split/train_aug.csv']
  df = pd.DataFrame()
  for file in files:
    data = pd.read_csv(file)
    df = pd.concat([df, data], axis=0)
  df.to_csv('/disk1/abtarget/dataset/new_split/train_aug.csv', index=False)'''

  '''dfs = pd.read_excel('/disk1/abtarget/dataset/Klebsiella.xlsx', sheet_name='Summary')
  name = set(dfs['sample'])
  df_fin = pd.DataFrame(columns =['name','VH', 'VL', 'target', 'label'])
  #df_list = []

  for inx, el in enumerate(name):
    df = dfs[dfs['sample'] == el]
    vl = df[df['type'] == 'light']['protein'].values[0]
    vh = df[df['type'] == 'heavy']['protein'].values[0]
    df_line = pd.DataFrame({'name': el, 'VH': vh, 'VL': vl, 'target': 'NonProtein', 'label': 1}, index = [inx])
    df_fin = pd.concat([df_fin, df_line])
    #df_fin = pd.concat([df_fin, df_line])
    #df_fin.append({'name': el, 'VH': vh, 'VL': vl, 'target': 'NonProtein', 'label': 1}, ignore_index=True)
  
  df_fin.to_csv('/disk1/abtarget/dataset/Klebsiella_test.csv', index=False)'''

  #data = pd.read_csv('/disk1/abtarget/dataset/sabdab_200423.txt', sep='/t')
  df = pd.read_csv('/disk1/abtarget/dataset/sabdab_200423.txt', sep='\t',usecols=[0,1,2,5], index_col=False)
  df_new = df.dropna()
  print(len(df_new))
  df_new = df_new[df_new['antigen_type'] != 'unknown']
  print(len(df_new))

  pdb_names = list(set(df_new['pdb']))
  print(len(pdb_names))
  df_nonrep = pd.DataFrame()
  output_path = '/disk1/abtarget/dataset/sabdab'

  protein_list = ['peptide | peptide | peptide', 'peptide | protein | protein', 'peptide | protein', 'protein', 'protein | peptide', 'protein | protein | protein | protein', 
                  'protein | peptide | protein', 'protein | protein', 'protein | protein | protein', 'peptide | peptide', 'peptide', 'protein | protein | protein | peptide',
                  'protein | protein | protein | protein | protein', 'protein | protein | peptide']
  
  hybrid_list = ['carbohydrate | protein', 'carbohydrate | protein | protein', 'protein | nucleic-acid']

  nonprotein_list = ['Hapten', 'carbohydrate', 'nucleic-acid', 'nucleic-acid | nucleic-acid | nucleic-acid', 'nucleic-acid | nucleic-acid']
  res = []
  non_seq = []

  '''for name in pdb_names:
    data = df_new[df_new['pdb'] == name].iloc[:1]
    df_nonrep = pd.concat([df_nonrep, data], axis=0)
    pdb_name = data['pdb'].to_string(index=False).lower()

    if data['antigen_type'].to_string(index=False) in protein_list:
      #pdb_file = __getPDB(data['pdb'].to_string(index=False), os.path.join(output_path,'protein'))
      pdb_path =  os.path.join(output_path, 'protein' , pdb_name + '.pdb')
      try:
        sequence_h = sequence(pdb_path, data['Hchain'].to_string(index=False), False)
        sequence_l = sequence(pdb_path, data['Lchain'].to_string(index=False), False)
        if len(sequence_h) == 0 or len(sequence_l) == 0:
          non_seq.append(data['pdb'].to_string(index=False).lower())
        else:
          res.append([pdb_name, sequence_h, sequence_l, data['antigen_type'].to_string(index=False), 0])
        
      except OSError as e:
        pass
    elif data['antigen_type'].to_string(index=False) in hybrid_list:
      pass
      #pdb_file = __getPDB(data['pdb'].to_string(index=False), os.path.join(output_path,'hybrid'))
    elif data['antigen_type'].to_string(index=False) in nonprotein_list:
      pdb_path =  os.path.join(output_path, 'nonprotein' , pdb_name + '.pdb')
      try:
        sequence_h = sequence(pdb_path, data['Hchain'].to_string(index=False), False)
        sequence_l = sequence(pdb_path, data['Lchain'].to_string(index=False), False)
        if len(sequence_h) == 0 or len(sequence_l) == 0:
          non_seq.append(data['pdb'].to_string(index=False).lower())
        else:
          res.append([pdb_name, sequence_h, sequence_l, data['antigen_type'].to_string(index=False), 1])
      except OSError as e:
        pass

      #pdb_file = __getPDB(data['pdb'].to_string(index=False), os.path.join(output_path,'nonprotein'))
  
  df = pd.DataFrame(res, columns = ['name', 'VH', 'VL', 'target', 'label'])
  #df.to_csv('/disk1/abtarget/dataset/sabdab_200423_train.csv', index = False)
  df.to_csv('/disk1/abtarget/dataset/sabdab_200423_train.csv', mode='a', header=False, index = False)

  df = pd.read_csv('/disk1/abtarget/dataset/sabdab_200423_train1.csv', header = None) '''

  '''for name in pdb_names:
    if name not in df['pdb']:

      num = len(df_new[df_new['pdb'] == name])
      for ind in range(num):
        data = df_new[df_new['pdb'] == name].iloc[ind:ind+1]

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
  #df.to_csv('/disk1/abtarget/dataset/sabdab_200423_train.csv', index = False)
  df.to_csv('/disk1/abtarget/dataset/sabdab_200423_train2.csv', header=False, index = False)'''

  '''#list_nonprotein = os.listdir('/disk1/abtarget/dataset/abdb/NR_LH_Protein_Kabat')
  list_nonprotein = [el.split('/')[-1].split('.')[0] for el in os.listdir('/disk1/abtarget/dataset/sabdab/nonprotein')]
  list_nonprotein.sort()
  dataset = pd.read_csv('/disk1/abtarget/dataset/sabdab_200423_train3.csv')
  list_pdb_dataset = list(dataset[dataset['label'] == 1]['name'].values)
  list_pdb_dataset.sort()
  num = 0
  for pdb_file in list_pdb_dataset:
    #target = pdb_file.split('/')[-1].split('_')[0].lower()
    #target = pdb_file.to_string(index=False)
    if pdb_file in list_nonprotein:
      list_nonprotein.remove(pdb_file)
  print(list_nonprotein)
  #print(len(list_nonprotein))'''

  '''dataset = pd.read_csv('/disk1/abtarget/dataset/abdb_dataset_old.csv')
  list_pdb_dataset = list(dataset[dataset['target'] == 'NonProtein']['name'].values)
  list_pdb_dataset.sort()
  df_recovery = pd.DataFrame()
  num = 0
  for pdb_file in list_pdb_dataset:
    target = pdb_file.split('/')[-1].split('_')[0].lower()
    #target = pdb_file.to_string(index=False)
    if target in list_nonprotein:
      print(pdb_file)
      data = dataset[dataset['name'] == pdb_file]['target'].values[0]
      print(data)
      num+=1
  print(num)'''

  '''res = []
  list_nonprotein = ['1h8s', '3gkz', '3gm0', '4gqp', '4h0i', '4lar', '4las', '5i4f', '5j74', '5j75', '6k4z', '6lfw', '6lfx', '6uy3', '7f35', '7rdm']
  pdb_path = os.path.join('/disk1/abtarget/dataset/sabdab/nonprotein', list_nonprotein[0]+'.pdb')
  sequence_h = sequence(pdb_path, 'A', False)
  sequence_l = sequence(pdb_path, 'B', False)
  sabdab = pd.read_csv('/disk1/abtarget/dataset/sabdab_200423.csv')
  res.append([list_nonprotein[0], sequence_h, sequence_l, sabdab[sabdab['pdb'] == list_nonprotein[0]]['antigen_type'].values[0], 1])
  print('A')'''

  #df_nonrep.to_csv('/disk1/abtarget/dataset/sabdab_200423_norep.csv', index=False)
  

  #element = set(df_new['antigen_type'])
  #print(element)
  #df_new.to_csv('/disk1/abtarget/dataset/sabdab_200423.csv', index=False)

  #read_file = pd.read_csv ('/disk1/abtarget/dataset/sabdab_200423.txt')
  #read_file.to_csv ('/disk1/abtarget/dataset/sabdab_200423.csv', index=None)

  #hybrid_df = pd.DataFrame(hybrid_dict, columns=['pdb', 'Hchain', 'Lchain', 'antigen_type'])  
  #hybrid_df.to_csv('/disk1/abtarget/dataset/sabdab_200423_statistics.csv', mode='a', header=False, index = False)

  #nonprotein_df = pd.DataFrame(nonprotein_dict, columns=['pdb', 'Hchain', 'Lchain', 'antigen_type'])  
  #nonprotein_df.to_csv('/disk1/abtarget/dataset/sabdab_200423_statistics.csv', mode='a', header=False, index = False)

  #df = pd.read_csv('/disk1/abtarget/dataset/sabdab_200423_train2.csv', header = None)
  #df.to_csv("/disk1/abtarget/dataset/sabdab_200423_train3.csv", header=['name', 'VH', 'VL', 'target', 'label'], index=False)

  print('Start')

  print('Original dimension')
  df = pd.read_csv ('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train_norep.csv')
  df_protein = df[df['label'] == 0]
  print(len(df_protein))
  df_nonprotein = df[df['label'] == 1]
  print(len(df_nonprotein))

  #df_protein = df_protein.drop_duplicates(subset=["VH", "VL"], keep='first')
  #print(len(df_protein))
  #df_nonprotein = df_nonprotein.drop_duplicates(subset=["VH", "VL"], keep='first')
  #print(len(df_nonprotein))

  print('Test')
  df_test = pd.read_csv ('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep.csv')
  df_protein_test = df_test[df_test['label'] == 0]
  print(len(df_protein_test))
  df_nonprotein_test = df_test[df_test['label'] == 1]
  print(len(df_nonprotein_test))

  print('Test + val')
  df_protein_train_val = df_protein[~df_protein['name'].isin(df_protein_test['name'])]
  print(len(df_protein_train_val))
  df_nonprotein_train_val = df_nonprotein[~df_nonprotein['name'].isin(df_nonprotein_test['name'])]
  print(len(df_nonprotein_train_val))

  print('Val')
  df_protein_val = df_protein_train_val.sample(n=50, random_state=42)
  print(len(df_protein_val))
  df_nonprotein_val = df_nonprotein.sample(n=50, random_state=42)
  print(len(df_nonprotein_val))

  print('Train')
  df_protein_train = df_protein_train_val[~df_protein_train_val['name'].isin(df_protein_val['name'])]
  print(len(df_protein_train))
  df_nonprotein_train = df_nonprotein_train_val[~df_nonprotein_train_val['name'].isin(df_nonprotein_val['name'])]
  print(len(df_nonprotein_train))


  df_protein_train.to_csv("/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep.csv", header=['name', 'VH', 'VL', 'target', 'label'], index=False)
  df_nonprotein_train.to_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep.csv', mode='a', header=False, index = False)

  df_protein_val.to_csv("/disk1/abtarget/dataset/sabdab/split/sabdab_200423_val_norep.csv", header=['name', 'VH', 'VL', 'target', 'label'], index=False)
  df_nonprotein_val.to_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_val_norep.csv', mode='a', header=False, index = False)





  print('Done')

  
    
