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

from src.protbert import Baseline
from src.baseline_dataset import CovAbDabDataset
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

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
          rap = len(classData1[0]) // len(classData1[1])+1
          trainList = rap * classData1[name]
          trainList = random.sample(trainList, len(classData1[0]))
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

def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, proportion=None):

  '''
  Split the dataset proportionally according to the sample label
  '''

  print('here')

  # Get classes
  classList = list(set(labels))
  resultList = {
    'test': [],
    'train': []
  }

  classData = {}
  for name in classList:

    # Get subsample of indexes for this class
    classData[name] = [ idx for idx, label in enumerate(labels) if label == name ]

  print('here')

  # Get shorter element
  shorter_class = min(classData.items(), key=lambda x: len(x[1]))[0]
  if proportion:
    subset_size = len(classData[shorter_class])
    for name in classList:
      if name == shorter_class:
        continue

      classData[name] = random.sample(classData[name], subset_size)

  print('here')

  classStats = {
    'train': {},
    'test': {}
  }

  print('here')

  for name in classList:
    train_size = round(len(classData[name]) * fraction)
    trainList = random.sample(classData[name], train_size)
    testList = [ idx for idx in classData[name] if idx not in trainList ]
    print(len(trainList))
    print(len(testList))

    # Update stats
    classStats['train'][name] = len(trainList)
    classStats['test'][name] = len(testList)

    # Concatenate indexes
    resultList['train'].extend(trainList)
    resultList['test'].extend(testList)

  print('here') 

  # Shuffle index lists
  for key in resultList:
    random.shuffle(resultList[key])
    print('{0} dataset:'.format(key))
    for name in classList:
      print(' Class {0}: {1}'.format(name, classStats[key][name]))

  print('here')

  # Construct the test and train datasets
  train_data = torch.utils.data.Subset(dataset, resultList['train'])
  test_data = torch.utils.data.Subset(dataset, resultList['test'])

  return train_data, test_data

def stratified_split_augontest(dataset : torch.utils.data.Dataset, labels, fraction, proportion=None):

  '''
  Split the dataset proportionally according to the sample label
  '''

  # Get classes
  classList = list(set(labels))
  resultList = {
    'test': [],
    'train': []
  }

  classData = {}
  for name in classList:

    # Get subsample of indexes for this class
    classData[name] = [ idx for idx, label in enumerate(labels) if label == name ]

  # Get shorter element
  shorter_class = min(classData.items(), key=lambda x: len(x[1]))[0]
  if proportion:
    subset_size = len(classData[shorter_class])
    for name in classList:
      if name == shorter_class:
        continue

      classData[name] = random.sample(classData[name], subset_size)

  classStats = {
    'train': {},
    'test': {}
  }
  for name in classList:
    pdb = [el for el in range(len(dataset)) if (dataset._data['name'].iloc[el].endswith('.pdb') and (dataset._data['label'].iloc[el] == name))]
    testList = random.sample(pdb, 60)
    trainList = [el for el in range(len(dataset)) if ((el not in testList) and (dataset._data['label'].iloc[el] == name))]

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
  train_data = torch.utils.data.Subset(dataset, resultList['train'])
  test_data = torch.utils.data.Subset(dataset, resultList['test'])

  return train_data, test_data


def controlled_split(dataset1 : torch.utils.data.Dataset, dataset2 : torch.utils.data.Dataset, labels1, labels2, subset, proportion, fraction):

  '''
  Split the dataset proportionally according to the sample label
  '''
  print(subset)

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

def stratified_split_kcross(dataset1 : torch.utils.data.Dataset, labels, subset, proportion, fraction):

  '''
  Split the dataset proportionally according to the sample label
  '''

  idx = np.asarray([ind for ind in range(len(labels))])
  print(idx.size)
  labels = np.asarray(labels)
  print(labels.size)


  kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

  for train_ix, test_ix in kfold.split(idx, labels):
    train_X, test_X = idx[train_ix], idx[test_ix]
    train_y, test_y = labels[train_ix], labels[test_ix]
    train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
    test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
    print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
  
    # Construct the test and train datasets
    train_data = torch.utils.data.Subset(dataset1, train_X)
    test_data = torch.utils.data.Subset(dataset1, test_X)

  return train_data, test_data 


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=1, save_folder=None, batch_size=8, device='cpu'):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  best_f1 = 0.0
  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []
  train_f1 = []
  test_f1 = []
  train_zero_class = []
  test_zero_class = []
  train_one_class = []
  test_one_class = []

  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("#" * 10)

    # Epoch train and validation phase
    for phase in ["train", "test"]:
      print("## " + f"{phase}".capitalize())
      if phase == "train":
        model.train()
      else:
        model.eval()

      actual = []
      pred = []

      # Iterate over the Data
      running_loss = 0.0
      running_correct = 0
      dataset_size = len(dataloaders[phase].dataset)
      size = len(dataloaders[phase])
      mcc_score = MCC()
      zeros = 0
      ones = 0

      for count, inputs in enumerate(dataloaders[phase]):

        labels = inputs['label'].to(device)
        actual.extend(labels.tolist())
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          pred.extend(preds.cpu().detach().numpy())
          one = torch.sum(preds).item()
          ones += one
          zeros += (args.batch_size - one)
          #print(outputs)
          #print(preds)
          #print(labels)
          loss = criterion(outputs, labels)
          if phase == "train":
            loss.backward()
            optimizer.step()

        # Stats
        current_loss = loss.item()
        running_loss += current_loss * len(inputs['label'])
        running_correct += torch.sum(preds == labels)
        '''if phase == "eval":
          scheduler.step(running_loss)'''
        
        #if phase == "train":
        #  scheduler.step()

        #weights = [zeros / dataset_size, ones / dataset_size]
        epoch_loss = running_loss / dataset_size
        #epoch_acc = running_correct.double() / dataset_size
        epoch_acc = balanced_accuracy_score(actual, pred)
        mcc = mcc_score.update(preds, labels)
        #epoch_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
        metric = BinaryF1Score().to(device)
        epoch_f1 = metric(preds, labels)
        #isPrint = True if count % 10 == 0 or count == size-1 else False
        isPrint = True if count == size-1 else False
        if isPrint:
          print('{phase} {count}/{total} Loss: {loss:.4f} Running Loss: {running_loss:.4f} Acc: {acc:.4f} MCC: {mcc:.4f} F1: {f1:.4f}'.format(
            total=size,
            count=count,
            phase=phase,
            running_loss=epoch_loss,
            loss=current_loss,
            acc=epoch_acc,
            mcc=mcc,
            f1 = epoch_f1
          ))

        # Deep copy the model & save checkpoint to file
        if phase == "test": 

          name = ''
        
          if epoch_acc > best_acc:

            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            name = '_best_accuracy'

          elif epoch_f1 > best_f1:

            best_f1 = epoch_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            name = '_best_f1' 


          if args.ensemble:
            save_path = os.path.join(save_folder, 'checkpoints', args.model, 'ensemble', str(subset))
          else:
            save_path = os.path.join(save_folder, 'checkpoints', args.model, 'single')
          
          if not os.path.exists(save_path):
            os.mkdir(save_path)
          
          checkpoint_path = os.path.join(save_path, args.save_name + name)
          torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "batch_size": batch_size,
            }, checkpoint_path)

      if phase == "train":
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc.item())
        train_zero_class.append(zeros)
        train_one_class.append(ones)
      else:
        test_loss.append(epoch_loss)
        test_acc.append(epoch_acc.item())
        test_zero_class.append(zeros)
        test_one_class.append(ones)

  # Store checkpoint
  
  checkpoint_path = os.path.join(save_path, 'epoch_{0}'.format(epoch+1))

  torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
    "batch_size": batch_size,
  }, checkpoint_path)

  time_elapsed = time.time() - since
  print('Training complete in {h}:{m}:{s}'.format(
    h=int(time_elapsed // 3600),
    m=int(time_elapsed // 60),
    s=int(time_elapsed % 60)
  ))
  print("Best test Acc: {0}".format(best_acc))

  # Load best model weights
  model.load_state_dict(best_model_wts)

  plot_train_test(train_loss, test_loss, 'Loss', 'train', 'val')
  plot_train_test(train_acc, test_acc, 'Accuracy', 'train', 'val')
  plot_train_test(train_zero_class, train_one_class, 'Train classes', 'zero', 'one', level = len(train_data))
  plot_train_test(test_zero_class, test_one_class, 'Test classes', 'zero', 'one', level = len(test_data))

  return model

def plot_train_test(train_list, test_list, title, label1, label2, level = None):
  epochs = [i  for i in range(args.epoch_number)]
  
  fig, ax = plt.subplots(figsize = (5, 2.7), layout = 'constrained')
  ax.plot(epochs, train_list, label = label1)
  ax.plot(epochs, test_list, label = label2)

  if level is not None:
    ax.plot(epochs, [level]*args.epoch_number, label = 'max')
    ax.plot(epochs, [level//2]*args.epoch_number, label = 'threshold')
    ax.plot(epochs, [0]*args.epoch_number, label = 'min')
    ax.set_ylabel('Classes')
  else:
    ax.set_ylabel(args.criterion)

  ax.set_xlabel('Epoch')
  #ax.set_ylabel(args.criterion)
  ax.set_title(title)
  ax.legend()

  image_save_path = os.path.join(args.checkpoint,'figures',args.save_name)

  if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)

  # Save figure
  plt.savefig(image_save_path+'/'+title +'.png')


if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser('Baseline for Abtarget classification', add_help=False) #, fromfile_prefix_chars="@")
  argparser.add_argument('-i', '--input', help='input model folder', type=str, default = "/disk1/abtarget/dataset")
  argparser.add_argument('-ch', '--checkpoint', help='checkpoint folder', type=str, default = "/disk1/abtarget")
  argparser.add_argument('-t', '--threads',  help='number of cpu threads', type=int, default=None)
  argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'protbert')
  argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=50)
  argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=16)
  argparser.add_argument('-r', '--random', type=int, help='Random seed', default=None)
  argparser.add_argument('-c', '--n_class', type=int, help='Number of classes', default=2)
  argparser.add_argument('-o', '--optimizer', type=str, help='Optimizer: SGD or Adam', default='Adam')
  argparser.add_argument('-l', '--lr', type=float, help='Learning rate', default=3e-5)
  argparser.add_argument('-cr', '--criterion', type=str, help='Criterion: BCE or Crossentropy', default='Crossentropy')
  argparser.add_argument('-en', '--ensemble', type=bool, help='Ensemble model', default= False)
  argparser.add_argument('-tr', '--pretrain', type=bool, help='Freeze encoder', default= True)
  argparser.add_argument('-sub', '--subset', type=int, help='Subset to train the model with', default = 0)
  argparser.add_argument('-tot', '--total', type=bool, help='Complete dataset', default = True)
  argparser.add_argument('-rep', '--repetition', type=bool, help='Repeat the non-protein class', default= False)
  argparser.add_argument('-aug', '--augmentation', type=bool, help='Augmentation of the non-protein class', default= True)

    

  # Parse arguments
  args = argparser.parse_args()
  if args.ensemble:
    args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', str(args.subset)])
  else:
    if args.total:
      args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', 'hallucination', 'rep', 'weightedLoss'])
      #args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', 'tot', 'lrscheduler', 'nodropout'])
    else:
      #args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', 'cross'])
      #args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', 'tot', 'hallucination', '13'])
      args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', 'tot', 'esm1b', 'no_weight'])


  print(f"Model: {args.model} | Epochs: {args.epoch_number} | Batch: {args.batch_size} | Optimizer: {args.optimizer} | Criterion: {args.criterion} | Learning rate: {args.lr}")
  
  # Set random seed for reproducibility
  if args.random:
    random.seed(args.random)
  
  # Create the dataset object
  #dataset1 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep_hal_10.csv')
  #dataset1 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep_hal_13.csv')
  dataset1 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep_aug_2.csv')
  #dataset1 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep.csv')
  #dataset1 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep.csv')
  dataset2 = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_val_norep.csv')
  

  if args.threads:
    torch.set_num_threads(args.threads)

  # Train test split 
  nn_train = 0.8
  save_path = os.path.join(args.input, 'checkpoints') 

  #train_data, test_data = controlled_split(dataset, dataset.labels, fraction=nn_train, subset = 0, proportion=0.5)

  if (args.ensemble):
    subset = args.subset
    train_data, test_data = controlled_split(dataset1, dataset2, dataset1.labels, dataset2.labels, fraction=nn_train, subset = subset, proportion=0.5)
  else:
    train_data, test_data = stratified_split1(dataset1, dataset2, dataset1.labels, dataset2.labels, tot = args.total, repetition = args.repetition)
    #subset = args.subset
    #train_data, test_data = stratified_split_kcross(dataset1, dataset1.labels, fraction=nn_train, subset = subset, proportion=0.5)
    print('Done')

    
  # Train and Test Dataloaders - (Wrap data with appropriate data loaders)
  train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
  )
  val_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True
  )

  # Set the device
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
  
  model = Baseline(args.batch_size, device, nn_classes=args.n_class, freeze_bert=args.pretrain, model_name=args.model)


  # Define criterion, optimizer and lr scheduler
  if args.criterion == 'Crossentropy':
    if args.total:
      if args.repetition is False and args.augmentation is False:
        #weights = [1, 2910/251] #[ 1 / number of instances for each class]
        #weights = [1, 2879/220] #[ 1 / number of instances for each class]
        weights = [1, 2874/215]
        #weights = [1, 2339/212]
        #weights = [1, 2339/1530]
        #weights = [1, 2824/2325]
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device) 
      else:
        criterion = torch.nn.CrossEntropyLoss().to(device) 
    else:
      criterion = torch.nn.CrossEntropyLoss().to(device) 
  else:
    criterion = torch.nn.BCELoss().to(device)

  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=1)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

  # Train model
  train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs=args.epoch_number,
    save_folder=args.checkpoint,
    batch_size=args.batch_size,
    device=device
  )

  print("\n ## Training DONE ")