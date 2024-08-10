import os
import time
import copy
import random
import argparse

import torch
from sklearn.svm import OneClassSVM, SVC
from sklearn.metrics import classification_report
import numpy as np
import umap
import seaborn as sns
from torchmetrics.classification import BinaryF1Score
from matplotlib import pyplot as plt

from src.protbert import BaselineOne
from src.baseline_dataset import SAbDabDataset
from src.metrics import MCC
from src.training_eval import final_score_eval, confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def embedding_phase(dataloaders, phase, model):

  print('embdedding phase')
  labels = []
  embeddings = []

  '''target =  {'peptide | peptide | peptide':0, 'peptide | protein | protein':1, 'peptide | protein':2, 'protein':3, 'protein | peptide':4, 'protein | protein | protein | protein':5, 
                  'protein | peptide | protein':6, 'protein | protein':7, 'protein | protein | protein':8, 'peptide | peptide':9, 'peptide':10, 'protein | protein | protein | peptide':11,
                  'protein | protein | protein | protein | protein':12, 'protein | protein | peptide':13,'Hapten':14, 'carbohydrate':15, 'nucleic-acid':16, 'nucleic-acid | nucleic-acid | nucleic-acid':17, 'nucleic-acid | nucleic-acid':18}'''
  

  for count, inputs in enumerate(dataloaders[phase]):

    labels.append(np.squeeze(inputs['label'].cpu().detach().numpy()))
    try:
      embeddings.append(np.squeeze(model(inputs).cpu().detach().numpy()))
    except:
      print('error')
  
  return labels, embeddings



def stratified_split1(dataset1 : torch.utils.data.Dataset, dataset2 : torch.utils.data.Dataset, dataset3 : torch.utils.data.Dataset, labels1, labels2, labels3, train_size, tot):

  '''
  Split the dataset proportionally according to the sample label
  '''

  # Get classes
  classList = list(set(labels1))
  resultList = {
    'train': [],
    'val': [],
    'test': []
  }

  classData1 = {}
  classData2 = {}
  classData3 = {}
  for name in classList:

    # Get subsample of indexes for this class
    classData1[name] = [ idx for idx, label in enumerate(labels1) if label == name ]
    classData2[name] = [ idx for idx, label in enumerate(labels2) if label == name ]
    classData3[name] = [ idx for idx, label in enumerate(labels3) if label == name ]

  classStats = {
    'train':{},
    'val':{},
    'test': {}
  }
  for name in classList:

    if tot:
      if name == 0:
        trainList = random.sample(classData1[name], 2824)
      else:
        trainList = classData1[name]
    else:
      if name == 0:
        trainList = random.sample(classData1[name], len(classData1[1]))
      else:
        trainList = classData1[name]

    #trainList = classData1[name]
    valList = classData2[name]
    testList = classData3[name]
    
    print(len(valList))
    print(len(trainList))

    # Update stats
    classStats['test'][name] = len(testList)
    classStats['val'][name] = len(valList)
    classStats['train'][name] = len(trainList)

    # Concatenate indexes
    resultList['test'].extend(testList)
    resultList['val'].extend(valList)
    resultList['train'].extend(trainList)
      

  train_data = torch.utils.data.Subset(dataset1, resultList['train'])
  val_data = torch.utils.data.Subset(dataset2, resultList['val'])
  test_data = torch.utils.data.Subset(dataset3, resultList['test'])

  return train_data, val_data, test_data


def controlled_split(dataset1 : torch.utils.data.Dataset, dataset2 : torch.utils.data.Dataset, dataset3 : torch.utils.data.Dataset, labels1, labels2, labels3, subset, proportion, fraction):

  '''
  Split the dataset proportionally according to the sample label
  '''
  print(subset)

  # Get classes
  classList = list(set(labels1))
  resultList = {
    'test': [],
    'val':[],
    'train': []
  }

  classData1 = {}
  classData2 = {}
  classData3 = {}

  for name in classList:
    # Get subsample of indexes for this class
    classData1[name] = [ idx for idx, label in enumerate(labels1) if label == name ]
    classData2[name] = [ idx for idx, label in enumerate(labels2) if label == name ]
    classData3[name] = [ idx for idx, label in enumerate(labels3) if label == name ]

  # Get shorter element
  shorter_class = min(classData1.items(), key=lambda x: len(x[1]))[0]
  if proportion:
    subset_size = len(classData1[shorter_class])

    '''for name in classList:
      if name == shorter_class:
        continue

      ## divide the class in subsets
      step = int(2 * subset_size/3)
      classDatasubset = [classData[name][base:base+subset_size-1] for base in range(0,len(classData[name]),step)]
      #classData[name] = random.sample(classData[name], subset_size)
      classData[name] =  classDatasubset[subset]'''

  classStats = {
    'train': {},
    'val':{},
    'test': {}
  }

  for name in classList:
    #train_size = round(subset_size * fraction)
    train_size = round(subset_size)
    
    if name == shorter_class:
      trainList = random.sample(classData1[name], train_size)
      #testList = [ idx for idx in classData[name] if idx not in trainList ]
    else:
      #testList = random.sample(classData1[name], len(classData1[shorter_class]) - train_size)
      trainList_tot = [ idx for idx in classData1[name]]
      random.shuffle(trainList_tot)
      step = int(2 * train_size/3)

      classDatasubset = []
      for base in range(0,len(trainList_tot),step):
        if base+train_size > len(trainList_tot):
          classDatasubset.append(trainList_tot[base:])
          break
        else:
          classDatasubset.append(trainList_tot[base:base+train_size])

      trainList =  classDatasubset[subset]
    valList = classData2[name]
    testList = classData3[name]
      

    # Update stats
    classStats['train'][name] = len(trainList)
    classStats['val'][name] = len(valList)
    classStats['test'][name] = len(testList)

    # Concatenate indexes
    resultList['train'].extend(trainList)
    resultList['val'].extend(valList)
    resultList['test'].extend(testList)

  # Shuffle index lists
  for key in resultList:
    random.shuffle(resultList[key])
    print('{0} dataset:'.format(key))
    for name in classList:
      print(' Class {0}: {1}'.format(name, classStats[key][name]))
  
  # Construct the test and train datasets
  train_data = torch.utils.data.Subset(dataset1, resultList['train'])
  val_data = torch.utils.data.Subset(dataset2, resultList['val'])
  test_data = torch.utils.data.Subset(dataset3, resultList['test'])

  return train_data, val_data, test_data

def controlled_split_kcross(dataset1 : torch.utils.data.Dataset, dataset2 : torch.utils.data.Dataset, labels1, labels2, subset, proportion, fraction):

  '''
  Split the dataset proportionally according to the sample label
  '''

  # Get classes
  classList = list(set(labels1))
  resultList = {
    'test': [],
    'train': []
  }

  classData1 = {}
  classData2 = {}

  for name in classList:
    # Get subsample of indexes for this class
    classData1[name] = [ idx for idx, label in enumerate(labels1) if label == name ]
    classData2[name] = [ idx for idx, label in enumerate(labels2) if label == name ]

  # Get shorter element
  shorter_class = min(classData1.items(), key=lambda x: len(x[1]))[0]
  if proportion:
    subset_size = len(classData1[shorter_class])

    '''for name in classList:
      if name == shorter_class:
        continue

      ## divide the class in subsets
      step = int(2 * subset_size/3)
      classDatasubset = [classData[name][base:base+subset_size-1] for base in range(0,len(classData[name]),step)]
      #classData[name] = random.sample(classData[name], subset_size)
      classData[name] =  classDatasubset[subset]'''

  classStats = {
    'train': {},
    'test': {}
  }

  for name in classList:
    #train_size = round(subset_size * fraction)
    train_size = round(subset_size)
    
    if name == shorter_class:
      trainList = random.sample(classData1[name], train_size)
      #testList = [ idx for idx in classData[name] if idx not in trainList ]
    else:
      #testList = random.sample(classData1[name], len(classData1[shorter_class]) - train_size)
      trainList_tot = [ idx for idx in classData1[name]]
      random.shuffle(trainList_tot)
      step = int(2 * train_size/3)

      classDatasubset = []
      for base in range(0,len(trainList_tot),step):
        if base+train_size > len(trainList_tot):
          classDatasubset.append(trainList_tot[base:])
          break
        else:
          classDatasubset.append(trainList_tot[base:base+train_size])

      trainList =  classDatasubset[subset]
      testList = classData2[name]
      

    # Update stats
    classStats['train'][name] = len(trainList)
    classStats['test'][name] = len(testList)

    # Concatenate indexes
    resultList['train'].extend(trainList)
    resultList['test'].extend(testList)

  # Shuffle index lists
  for key in resultList:
    random.shuffle(resultList[key])
    print('{0} dataset:'.format(key))
    for name in classList:
      print(' Class {0}: {1}'.format(name, classStats[key][name]))
  
  # Construct the test and train datasets
  train_data = torch.utils.data.Subset(dataset1, resultList['train'])
  test_data = torch.utils.data.Subset(dataset2, resultList['test'])

      
  # Save validation split in a txt file
  #with open('/disk1/abtarget/dataset/split/test.txt','w') as file:
  #  file.write("\n".join(str(item) for item in resultList['test']))
  #  #data.write(str(dictionary))

  return train_data, test_data



def eval(classifier, name, args):
  print('Start Eval - embedding')
  labels, embeddings =  embedding_phase(dataloaders, name, model)

  prediction = classifier.predict(embeddings)

  if args.ocsvm:
    prediction = np.where(prediction == -1, 1, np.where(prediction == 1, 0, prediction))
  
  if args.lasso:
    prediction =  prediction.round()

  return labels, prediction



if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser('Baseline for Abtarget classification', add_help=False) #, fromfile_prefix_chars="@")
  argparser.add_argument('-i', '--input', help='input model folder', type=str, default = "/disk1/abtarget/dataset")
  argparser.add_argument('-ch', '--checkpoint', help='checkpoint folder', type=str, default = "/disk1/abtarget")
  argparser.add_argument('-t', '--threads',  help='number of cpu threads', type=int, default=None)
  argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'antiberty')
  argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=50)
  argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=1)
  argparser.add_argument('-r', '--random', type=int, help='Random seed', default=None)
  argparser.add_argument('-c', '--n_class', type=int, help='Number of classes', default=2)
  argparser.add_argument('-o', '--optimizer', type=str, help='Optimizer: SGD or Adam', default='Adam')
  argparser.add_argument('-l', '--lr', type=float, help='Learning rate', default=3e-5)
  argparser.add_argument('-cr', '--criterion', type=str, help='Criterion: BCE or Crossentropy', default='Crossentropy')
  argparser.add_argument('-en', '--ensemble', type=bool, help='Ensemble model', default= True)
  argparser.add_argument('-tr', '--pretrain', type=bool, help='Freeze encoder', default= True)
  argparser.add_argument('-sub', '--subset', type=int, help='Subset to train the model with', default = 7)
  argparser.add_argument('-train', '--training', type=str, help='Training dataset', default = "/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep.csv")
  argparser.add_argument('-val', '--val', type=str, help='Validation dataset', default = "/disk1/abtarget/dataset/sabdab/split/sabdab_200423_val_norep.csv")
  argparser.add_argument('-test', '--test', type=str, help='Test dataset', default = "/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep.csv")
  argparser.add_argument('-lasso', '--lasso', type=bool, help='LASSO classification', default = False)
  argparser.add_argument('-rf', '--rf', type=bool, help='Random Forest classification', default = False)
  argparser.add_argument('-knn', '--knn', type=bool, help='KNN classification', default = False)
  argparser.add_argument('-ocsvm', '--ocsvm', type=bool, help='OCSVM classification', default = True)


    

  # Parse arguments
  args = argparser.parse_args()

  if args.ensemble:
    args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'new_split', 'norep', str(args.subset)])
  else:
    args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', '7_2_1', 'norep', '512'])

  print(f"Model: {args.model} | Epochs: {args.epoch_number} | Batch: {args.batch_size} | Optimizer: {args.optimizer} | Criterion: {args.criterion} | Learning rate: {args.lr}")
  
  # Set random seed for reproducibility
  if args.random:
    random.seed(args.random)
  
  # Create the dataset object
  #dataset =  SAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train_norep.csv')

  dataset1 = SAbDabDataset(args.training)
  dataset2 = SAbDabDataset(args.val)
  dataset3 = SAbDabDataset(args.test)
  

  if args.threads:
    torch.set_num_threads(args.threads)

  # Train test split 
  nn_train = 0.8
  save_path = os.path.join(args.input, 'checkpoints') 

  #train_data, test_data = controlled_split(dataset, dataset.labels, fraction=nn_train, subset = 0, proportion=0.5)

  if (args.ensemble):
    subset = args.subset
    train_data, val_data, test_data = controlled_split(dataset1, dataset2, dataset3, dataset1.labels, dataset2.labels,  dataset3.labels, fraction=nn_train, subset = subset, proportion=0.5)
  else:
    train_data, val_data, test_data = stratified_split1(dataset1, dataset2, dataset3, dataset1.labels, dataset2.labels,  dataset3.labels, train_size=10000, tot = True)
    
    
  # Train and Test Dataloaders - (Wrap data with appropriate data loaders)
  train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
  )
  val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True
  )
  test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True
  )

  # Set the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device {0}'.format(device))

  # Train script
  dataloaders = {
    "train": train_loader,
    "val": val_loader, 
    "test": test_loader
  }

  # Select model
  model_name = args.model.lower()
  print(model_name)
  model = BaselineOne(args.batch_size, device, nn_classes=args.n_class, freeze_bert=args.pretrain, model_name=args.model)

  model.eval()

  labels, embeddings =  embedding_phase(dataloaders, "train", model)

  if args.lasso:
    classifier = Lasso(alpha = 0.1).fit(embeddings,labels)
  elif args.knn:
    k = 5
    classifier = KNeighborsClassifier(n_neighbors=k, weights='distance').fit(embeddings,labels)
  elif args.rf:
    estimators = 100
    classifier = RandomForestClassifier(n_estimators=estimators, criterion="entropy").fit(embeddings,labels)
  else:
    if args.ensemble:
      nu = 0.5
    else:
      nu = 0.07
    classifier = OneClassSVM(nu = nu, kernel = 'rbf', gamma = 'auto').fit(embeddings)

  
  print('Training')
  prediction = classifier.predict(embeddings)
  if args.lasso: 
    prediction = prediction.round()
  if args.ocsvm:
    prediction = np.where(prediction == -1, 1, np.where(prediction == 1, 0, prediction))
  
  final_score_eval(labels, prediction)

  print('Validation')
  labels, predictions = eval(classifier, 'val', args)
  final_score_eval(labels, predictions)

  print('Test')
  labels, predictions = eval(classifier, 'test', args)
  final_score_eval(labels, predictions)

  #confusion_matrix(labels, predictions)
  