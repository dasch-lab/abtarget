import os
import time
import copy
import random
import argparse
import math
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import random_split

from src.protbert import Baseline
from src.baseline_dataset import MabMultiStrainBinding, CovAbDabDataset
from src.metrics import MCC
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
random.seed(22)

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

def load_data(dataset : torch.utils.data.Dataset, labels, fraction, proportion=None):

  '''
  Split the dataset proportionally according to the sample label
  '''

  # Get classes
  classList = list(set(labels))
  resultList = {
    'test': []
  }

  classData = {}
  for name in classList:

    # Get subsample of indexes for this class
    classData[name] = [ idx for idx, label in enumerate(labels) if label == name ]

  classStats = {
    'test': {}
  }
  for name in classList:

    testList = classData[name]

    # Update stats
    classStats['test'][name] = len(testList)

    # Concatenate indexes
    resultList['test'].extend(testList)

  # Shuffle index lists
  for key in resultList:
    random.shuffle(resultList[key])
    print('{0} dataset:'.format(key))
    for name in classList:
      print(' Class {0}: {1}'.format(name, classStats[key][name]))

  test_data = torch.utils.data.Subset(dataset, resultList['test'])

  return test_data

def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, proportion=None):

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
    train_size = round(len(classData[name]) * fraction)
    trainList = random.sample(classData[name], train_size)
    testList = [ idx for idx in classData[name] if idx not in trainList ]

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
  name = [dataset._data['name'].iloc[ind] for ind in test_data.indices]

  return train_data, test_data, name

def stratified_split1(dataset1 : torch.utils.data.Dataset, dataset2 : torch.utils.data.Dataset, labels1, labels2, train_size):

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

    '''if name == 0:
      trainList = random.sample(classData1[name], train_size)
    else:
      trainList = classData1[name]'''

    testList = classData2[name]
    trainList = classData1[name]
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

def eval_model(model, dataloaders):
  origin = []
  pred = []
  
  model.eval()
  for count, inputs in enumerate(dataloaders["test"]):
    labels = inputs['label'].to(device)
    origin.extend(labels.cpu().detach().numpy())
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    pred.extend(preds.cpu().detach().numpy())

  confusion_matrix = metrics.confusion_matrix(np.asarray(origin), np.asarray(pred))
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
  cm_display.plot()
  plt.show()
  #plt.matshow(confusion_matrix)
  #plt.title('Confusion Matrix')
  #plt.colorbar()
  #plt.ylabel('True Label')
  #plt.xlabel('Predicated Label')
  plt.savefig('confusion_matrix.jpg')


  return origin, pred

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
  ax.legend();

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
  argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=200)
  argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=16)
  argparser.add_argument('-r', '--random', type=int, help='Random seed', default=None)
  argparser.add_argument('-c', '--n_class', type=int, help='Number of classes', default=2)
  argparser.add_argument('-o', '--optimizer', type=str, help='Optimizer: SGD or Adam', default='SGD')
  argparser.add_argument('-l', '--lr', type=float, help='Learning rate', default=3e-5)
  argparser.add_argument('-cr', '--criterion', type=str, help='Criterion: BCE or Crossentropy', default='Crossentropy')

  # Parse arguments
  args = argparser.parse_args()
  args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion])

  print(f"Model: {args.model} | Epochs: {args.epoch_number} | Batch: {args.batch_size} | Optimizer: {args.optimizer} | Criterion: {args.criterion} | Learning rate: {args.lr}")
  
  # Set random seed for reproducibility
  if args.random:
    random.seed(22)
  
  # Create the dataset object
  dataset = CovAbDabDataset('/disk1/abtarget/dataset/split/test.csv')
  #dataset = CovAbDabDataset('/disk1/abtarget/dataset/split/aug_test.csv')
  #dataset = CovAbDabDataset('/disk1/abtarget/dataset/split/train_aug.csv')

  dataset1 = CovAbDabDataset('/disk1/abtarget/dataset/split/train.csv')
  dataset2 = CovAbDabDataset('/disk1/abtarget/dataset/split/test.csv')
  
  

  if args.threads:
    torch.set_num_threads(args.threads)

  # Train test split 
  nn_train = 0.8
  #test_data = load_data(dataset, dataset.labels, fraction=nn_train, proportion=0.5)
  #train_data, test_data = stratified_split_augontest(dataset, dataset.labels, fraction=nn_train, proportion=0.5)
  #train_data, test_data, name = stratified_split(dataset, dataset.labels, fraction=nn_train, proportion=0.5)
  train_data, test_data = stratified_split1(dataset1, dataset2, dataset1.labels, dataset2.labels, train_size = 10000)
    

  # Save Dataset or Dataloader for later evaluation
  save_dataset = True
  if save_dataset:
    save_path = os.path.join(args.input, 'checkpoints')
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    
  val_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True
  )

  # Set the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device {0}'.format(device))

  # Train script
  dataloaders = {
    "test": val_loader
  }

  # Select model
  model = None
  model_name = args.model.lower()
  model = Baseline(args.batch_size, device, nn_classes=args.n_class, freeze_bert=True, model_name=args.model) 


  if args.criterion == 'Crossentropy':
    criterion = torch.nn.CrossEntropyLoss().to(device) #TODO
  else:
    criterion = torch.nn.BCELoss().to(device)

  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=1)

  checkpoint = torch.load('/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_noaugval_accuracy_F1_256_best_f1')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']

  # Train model
  pred, org = eval_model(
    model,
    dataloaders
  )

  #df = pd.DataFrame(list(zip(name, pred, org)), columns=['Name','GT','Pred'])
  #df.to_csv('/disk1/abtarget/dataset/split/res_val.csv', index = False)



  print("\n ## Training DONE ")