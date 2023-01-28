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

  return train_data, test_data

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=1, save_folder=None, batch_size=8, device='cpu'):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []
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
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
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
        if phase == "train":
          scheduler.step()

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_correct.double() / dataset_size
        mcc = mcc_score.update(preds, labels)
        #isPrint = True if count % 10 == 0 or count == size-1 else False
        isPrint = True if count == size-1 else False
        if isPrint:
          print('{phase} {count}/{total} Loss: {loss:.4f} Running Loss: {running_loss:.4f} Acc: {acc:.4f} MCC: {mcc:.4f}'.format(
            total=size,
            count=count,
            phase=phase,
            running_loss=epoch_loss,
            loss=current_loss,
            acc=epoch_acc,
            mcc=mcc
          ))

        # Deep copy the model & save checkpoint to file
        if phase == "test" and epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model_wts = copy.deepcopy(model.state_dict())

          save_path = os.path.join(save_folder, 'checkpoints')
          if not os.path.exists(save_path):
            os.mkdir(save_path)

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

  plot_train_test(train_loss, test_loss, 'Loss', 'train', 'test')
  plot_train_test(train_acc, test_acc, 'Accuracy', 'train', 'test')
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
  ax.legend();

  # Save figure
  plt.savefig('/disk1/abtarget/figures/'+title +'.png')




if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser('Baseline for Abtarget classification', add_help=False) #, fromfile_prefix_chars="@")
  argparser.add_argument('-i', '--input', help='input model folder', type=str, default = "/disk1/abtarget/dataset")
  argparser.add_argument('-ch', '--checkpoint', help='checkpoint folder', type=str, default = "/disk1/abtarget")
  argparser.add_argument('-t', '--threads',  help='number of cpu threads', type=int, default=None)
  argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'protbert')
  argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=100)
  argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=16)
  argparser.add_argument('-r', '--random', type=int, help='Random seed', default=None)
  argparser.add_argument('-c', '--n_class', type=int, help='Number of classes', default=2)
  argparser.add_argument('-o', '--optimizer', type=str, help='Optimizer: SGD or Adam', default='Adam')
  argparser.add_argument('-l', '--lr', type=float, help='Learning rate', default=1e-6)
  argparser.add_argument('-cr', '--criterion', type=str, help='Criterion: BCE or Crossentropy', default='Crossentropy')

  # Parse arguments
  args = argparser.parse_args()

  print(f"Model: {args.model} | Epochs: {args.epoch_number} | Batch: {args.batch_size} | Optimizer: {args.optimizer} | Criterion: {args.criterion} | Learning rate: {args.lr}")
  
  # Set random seed for reproducibility
  if args.random:
    random.seed(args.random)
  
  # Create the dataset object
  dataset = CovAbDabDataset(os.path.join(args.input, 'abdb_dataset.csv'))
  

  if args.threads:
    torch.set_num_threads(args.threads)

  # Train test split 
  nn_train = 0.8
  train_data, test_data = stratified_split(dataset, dataset.labels, fraction=nn_train, proportion=0.5)

  # Save Dataset or Dataloader for later evaluation
  save_dataset = True
  if save_dataset:
    save_path = os.path.join(args.input, 'checkpoints')
    if not os.path.exists(save_path):
      os.mkdir(save_path)

    # Store datasets
    torch.save(train_data, os.path.join(save_path, 'train_data.pt'))
    torch.save(test_data, os.path.join(save_path, 'test_data.pt'))
    
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
  #if model_name == 'rcnn':
  #  hidden_size = 256
  #  output_size = 2
  model = Baseline(args.batch_size, device, nn_classes=args.n_class, freeze_bert=True) 

  #if model == None:
  #  raise Exception('Unable to initialize model \'{model}\''.format(model_name))

  # Define criterion, optimizer and lr scheduler
  if args.criterion == 'Crossentropy':
    criterion = torch.nn.CrossEntropyLoss().to(device) #TODO
  else:
    criterion = torch.nn.BCELoss().to(device)

  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoch_number//4, gamma=1)

  # Train model
  train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    exp_lr_scheduler,
    num_epochs=args.epoch_number,
    save_folder=args.checkpoint,
    batch_size=args.batch_size,
    device=device
  )

  print("\n ## Training DONE ")