from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter
import os
import time
import copy
import random
import argparse
import math
from collections import defaultdict
import sys

import pandas as pd
import torch
from torch.utils.data import random_split

from src.protbert import Baseline, BaselineOne, BaselineVHVL
from src.baseline_dataset import MabMultiStrainBinding, CovAbDabDataset, VHVLDataset
from src.metrics import MCC
from matplotlib import pyplot as plt

from sklearn.svm import OneClassSVM, SVC
from sklearn.metrics import classification_report
random.seed(42)
import numpy as np
from sklearn import metrics
import umap
import seaborn as sns
from sklearn.metrics import f1_score
from torchmetrics.classification import BinaryF1Score
import csv
import pandas as pd

def stratified_split1(dataset1 : torch.utils.data.Dataset, dataset2 : torch.utils.data.Dataset, labels1, labels2, tot, repetition):

  '''
  Split the dataset proportionally according to the sample label
  '''

  # Get classes
  classList = list(set(labels1))
  resultList = {
    'train': [],
    'test': []
  }

  classData1 = {}
  classData2 = {}
  for name in classList:

    # Get subsample of indexes for this class
    classData1[name] = [ idx for idx, label in enumerate(labels1) if label == name ]
    classData2[name] = [ idx for idx, label in enumerate(labels2) if label == name ]

  classStats = {
    'train':{},
    'test': {}
  }
  for name in classList:

    if tot:
      trainList = classData1[name]
      if repetition:
        if name == 1:
          rap = len(classData1[0]) // len(classData1[1])
          trainList = rap * classData1[name]
    else:
      #if name == 0:
      #  trainList = random.sample(classData1[name], len(classData1[1]))
      if name == 1:
        trainList = random.sample(classData1[name], len(classData1[0]))
      else:
        trainList = classData1[name]

    #trainList = classData1[name]
    testList = classData2[name]
    
    print(len(trainList))

    # Update stats
    classStats['test'][name] = len(testList)
    classStats['train'][name] = len(trainList)

    # Concatenate indexes
    resultList['test'].extend(testList)
    resultList['train'].extend(trainList)

  # Shuffle index lists
  '''for key in resultList:
    random.shuffle(resultList[key])
    print('{0} dataset:'.format(key))
    for name in classList:
      print(' Class {0}: {1}'.format(name, classStats[key][name]))'''
      

  train_data = torch.utils.data.Subset(dataset1, resultList['train'])
  test_data = torch.utils.data.Subset(dataset2, resultList['test'])

  return train_data, test_data


def stratified_split2(dataset1 : torch.utils.data.Dataset, dataset2 : torch.utils.data.Dataset, labels1, labels2, tot, repetition):

  '''
  Split the dataset proportionally according to the sample label
  '''

  # Get classes
  classList = list(set(labels1))
  resultList = {
    'train': [],
    'test': []
  }

  classData1 = {}
  classData2 = {}
  for name in classList:

    # Get subsample of indexes for this class
    classData1[name] = [ idx for idx, label in enumerate(labels1) if label == name ]
    classData2[name] = [ idx for idx, label in enumerate(labels2) if label == name ]

  classStats = {
    'train':{},
    'test': {}
  }
  for name in classList:

    if tot:
      trainList = classData1[name]
      if repetition:
        if name == 1:
          rap = len(classData1[0]) // len(classData1[1])
          trainList = rap * classData1[name]
    else:
      #if name == 0:
      #  trainList = random.sample(classData1[name], len(classData1[1]))
      if name == 1:
        trainList = random.sample(classData1[name], len(classData1[0]))
      else:
        trainList = classData1[name]

    #trainList = classData1[name]
    testList = classData2[name]
    
    print(len(trainList))

    # Update stats
    classStats['test'][name] = len(testList)
    classStats['train'][name] = len(trainList)

    # Concatenate indexes
    resultList['test'].extend(testList)
    resultList['train'].extend(trainList)

  # Shuffle index lists
  '''for key in resultList:
    random.shuffle(resultList[key])
    print('{0} dataset:'.format(key))
    for name in classList:
      print(' Class {0}: {1}'.format(name, classStats[key][name]))'''
      

  train_data = torch.utils.data.Subset(dataset1, resultList['train'])
  test_data = torch.utils.data.Subset(dataset2, resultList['test'])

  return train_data, test_data


def embedding_phase(dataloaders, phase):

  print('embdedding phase')
  labels = []
  embeddings = []

  for count, inputs in enumerate(dataloaders[phase]):
    #labels.append(target[inputs['label'].cpu().detach()])
    try:
      embeddings.append(np.squeeze(model(inputs).cpu().detach().numpy()))
      labels.append(np.squeeze(inputs['label'].cpu().detach().numpy()))
    except:
      print('error')
    
  return labels, embeddings
    


if __name__ == "__main__":

    # Initialize the argument parser
    argparser = argparse.ArgumentParser('Baseline for Abtarget classification', add_help=False) #, fromfile_prefix_chars="@")
    argparser.add_argument('-i', '--input', help='input model folder', type=str, default = "/disk1/abtarget/dataset")
    argparser.add_argument('-ch', '--checkpoint', help='checkpoint folder', type=str, default = "/disk1/abtarget")
    argparser.add_argument('-t', '--threads',  help='number of cpu threads', type=int, default=None)
    argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'protbert')
    argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=50)
    argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=1)
    argparser.add_argument('-r', '--random', type=int, help='Random seed', default=None)
    argparser.add_argument('-c', '--n_class', type=int, help='Number of classes', default=2)
    argparser.add_argument('-o', '--optimizer', type=str, help='Optimizer: SGD or Adam', default='Adam')
    argparser.add_argument('-l', '--lr', type=float, help='Learning rate', default=3e-5)
    argparser.add_argument('-cr', '--criterion', type=str, help='Criterion: BCE or Crossentropy', default='Crossentropy')
    argparser.add_argument('-en', '--ensemble', type=bool, help='Ensemble model', default= False)
    argparser.add_argument('-tr', '--pretrain', type=bool, help='Freeze encoder', default= True)
    argparser.add_argument('-sub', '--subset', type=int, help='Subset to train the model with', default = 7)

    # Parse arguments
    args = argparser.parse_args()
    nn_train = 0.8

    #dataset1 = VHVLDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep.csv')
    #dataset2 = VHVLDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep.csv')
    dataset1 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep.csv')
    dataset2 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep.csv')
    train_data, test_data = stratified_split1(dataset1, dataset2, dataset1.labels, dataset2.labels, tot = True, repetition=False)


    # Train and Test Dataloaders - (Wrap data with appropriate data loaders)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, num_workers=0, pin_memory=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device {0}'.format(device))

    # Train script
    dataloaders = {
        "train": train_loader, 
        "test": val_loader
    }

    # Select model
    model = None
    model_name = args.model.lower()
    print(model_name)

    model = BaselineOne(args.batch_size, device, nn_classes=args.n_class, freeze_bert=args.pretrain, model_name=args.model) 
    #model = BaselineVHVL(args.batch_size, device, nn_classes=args.n_class, freeze_bert=args.pretrain, model_name=args.model) 

    labels, embeddings =  embedding_phase(dataloaders, "train")

    X = embeddings
    y = labels

    # TODO create the embeddings and save them to perform SMOTE otherwise

    #oversample = SMOTE()
    #X,y = oversample.fit_resample(X,y)

    #d = {'X':X, 'y':y}
    #df = pd.DataFrame(d, columns = ['X', 'y'])
    #df.to_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep_protbert_SMOTE.csv',header = True, index=False)

    #counter = Counter(y)
    #print(counter)

    X = [el.tolist() for el in embeddings]
    y = [el.tolist() for el in labels]

    d = {'X':X, 'y':y}
    df = pd.DataFrame(d, columns = ['X', 'y'])
    df.to_csv('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep_protbert_embeddings_SMOTE.csv',header = True, index=False)
