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

from src.protbert import Baseline, MLP
from src.baseline_dataset import CovAbDabDataset, SMOTEDataset, MyDataset
from src.metrics import MCC
from src.data_loading_split import stratified_split
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
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')



def embedding_phase(dataloaders, phase):

  print('embdedding phase')
  labels = []
  embeddings = []

  for count, inputs in enumerate(dataloaders[phase]):

    labels.append(np.squeeze(inputs['label'].cpu().detach().numpy()))
    #labels.append(target[inputs['label'].cpu().detach()])
    try:
      embeddings.append(np.squeeze(model(inputs).cpu().detach().numpy()))
      return labels, embeddings
    except:
      print('error')


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

      print(len(dataloaders[phase]))
      try:
        for count, inputs in enumerate(dataloaders[phase]):

          labels = inputs['label'].to(device)
          if args.smote:
            inputs = inputs['X'].to(device)
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
          running_loss += current_loss * len(labels)
          running_correct += torch.sum(preds.view(-1) == labels.view(-1))
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

            '''elif epoch_f1 > best_f1:

              best_f1 = epoch_f1
              best_model_wts = copy.deepcopy(model.state_dict())
              name = '_best_f1' '''


            if args.ensemble:
              save_path = os.path.join(save_folder, 'checkpoints', args.model, 'ensemble', str(subset))
            else:
              save_path = os.path.join(save_folder, 'checkpoints', args.model, 'single')
            
            if not os.path.exists(save_path):
              os.mkdir(save_path)
            
            checkpoint_path = os.path.join(save_path, args.save_name + name)
            '''torch.save({
              "epoch": epoch,
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "loss": loss,
              "batch_size": batch_size,
              }, checkpoint_path)'''

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
      except:
        print('error')

  '''# Store checkpoint
  
  checkpoint_path = os.path.join(save_path, 'epoch_{0}'.format(epoch+1))

  torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
    "batch_size": batch_size,
  }, checkpoint_path)'''

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

  return model, checkpoint_path

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

def eval_model(model, dataloaders):
  origin = []
  pred = []
  misclassified =[]
  
  model.eval()
  for count, inputs in enumerate(dataloaders["test"]):
    labels = inputs['label'].to(device)
    if args.smote:
      inputs = inputs['X'].to(device)
    origin.extend(labels.cpu().detach().numpy())
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    pred.extend(preds.cpu().detach().numpy())
    #if preds.cpu().detach().numpy() != labels.cpu().detach().numpy():
    #  misclassified.append([inputs['name'][0], inputs['target'][0], labels.cpu().numpy()[0], preds.cpu().detach().numpy()[0]])
    
  #print(misclassified)

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

def Average(lst): 
    return sum(lst) / len(lst) 


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
  argparser.add_argument('-smote', '--smote', type=bool, help='SMOTE augmentation', default= False)
  argparser.add_argument('-hal', '--hallucination', type=bool, help='FvHallucinator augmentation', default= True)
  argparser.add_argument('-esm', '--esm', type=bool, help='ESM augmentation', default= False)
  argparser.add_argument('-double', '--double', type=bool, help='Double dataset', default= False)

  precision = []
  recall = []
  f1 = []
  accuracy = []
  mcc = []

    
  for i in range(20):

    print(f"Subset {i}")

    # Parse arguments
    args = argparser.parse_args()
    if args.ensemble:
      args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', str(args.subset)])
    else:
      if args.total:
        args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'bootstrap', 'esm2'])
      else:
        args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', 'single', str(i)])


    print(f"Model: {args.model} | Epochs: {args.epoch_number} | Batch: {args.batch_size} | Optimizer: {args.optimizer} | Criterion: {args.criterion} | Learning rate: {args.lr}")
    
    # Set random seed for reproducibility
    if args.random:
      random.seed(args.random)
    
    # Create the dataset object
    if args.smote:
      if args.model == 'antiberty':
        dataset = SMOTEDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep_antiberty_embeddings_SMOTE.csv')
      elif args.model == 'protbert':
        dataset = SMOTEDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep_protbert_embeddings_SMOTE.csv')
      else:
        exit()
    elif args.hallucination:
      dataset = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep_hallucination.csv')
    elif args.esm:
      dataset = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep_esm.csv')
    else:
      dataset = CovAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep.csv')
    

    if args.threads:
      torch.set_num_threads(args.threads)

    # Train test split 
    nn_train = 0.8
    save_path = os.path.join(args.input, 'checkpoints') 

    #train_data, test_data = controlled_split(dataset, dataset.labels, fraction=nn_train, subset = 0, proportion=0.5)

    if (args.ensemble):
      subset = args.subset
      train_data, val_data, test_data = stratified_split(dataset, labels = dataset.labels, step = i, subset_size=50, rep = args.repetition, tot = args.total)
    elif(args.smote):
      train_data, val_data, test_data = stratified_split(dataset, labels = dataset.labels, step = i, subset_size=50, rep = args.repetition, tot = args.total)
      
      embeddings_train = torch.stack([sample["X"] for i, sample in enumerate(train_data) if i in train_data.indices])
      labels_train = torch.stack([sample["label"] for i, sample in enumerate(train_data) if i in train_data.indices])

      oversample = SMOTE()
      embeddings_train, labels_train = oversample.fit_resample(embeddings_train,labels_train)
      embeddings_train = torch.tensor(embeddings_train).clone().detach()
      labels_train = torch.tensor(labels_train).clone().detach()
      my_dataset = MyDataset(embeddings_train, labels_train)
      classList = list(set(labels_train))
      resultList = {'train': []}
      resultList['train'] = [idx for idx, label in enumerate(labels_train) if label in classList]
      train_data = torch.utils.data.Subset(my_dataset, resultList['train'])
    
    elif(args.hallucination or args.esm):
      train_data, val_data, test_data = stratified_split(dataset, labels = dataset.labels, names = dataset.name, hallucination = args.hallucination, esm = args.esm, double = args.double, step = i, subset_size=50, rep = args.repetition, tot = args.total)
    else:
      train_data, val_data, test_data = stratified_split(dataset, labels = dataset.labels, step = i, subset_size=50, rep = args.repetition, tot = args.total)

      
    # Train and Test Dataloaders - (Wrap data with appropriate data loaders)
    train_loader = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
      val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
      test_data, batch_size=1, num_workers=4, pin_memory=True
    )

    dataloaders = {
      "train": train_loader, 
      "val": val_loader,
      "test": test_loader
      }

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device {0}'.format(device))

    # Select model
    model = None
    model_name = args.model.lower()
    print(model_name)

    if args.smote:
      model = MLP(args.batch_size, device, nn_classes=args.n_class, freeze_bert=args.pretrain, model_name=args.model)
    else:
      model = Baseline(args.batch_size, device, nn_classes=args.n_class, freeze_bert=args.pretrain, model_name=args.model)


    # Define criterion, optimizer and lr scheduler
    if args.criterion == 'Crossentropy':
      if args.total:
        if args.repetition is False and args.augmentation is False:
          weights = [1, 2874/215] #[ 1 / number of instances for each class]
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
    model, checkpoint_path = train_model(
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

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    pred, org = eval_model(
    model,
    dataloaders
    )

    precision.append(metrics.precision_score(org, pred))
    recall.append(metrics.recall_score(org, pred))
    f1.append(metrics.f1_score(org, pred))
    accuracy.append(metrics.accuracy_score(org, pred))
    mcc.append(metrics.matthews_corrcoef(org, pred))

    print('Precision: ', precision[i])
    print('Recall: ', recall[i])
    print('F1: ', f1[i])
    print('Accuracy: ', accuracy[i])
    print('MCC: ', mcc[i])
  
  print('Precision: ', precision)
  print('Recall: ', recall)
  print('F1: ', f1)
  print('Accuracy: ', accuracy)
  print('MCC: ', mcc)

  print('First 10 bootstraps')
  print(f'Precision: Mean - {Average(precision[:10])} | Std - {sum([((x - Average(precision[:10])) ** 2) for x in precision[:10]]) / len(precision[:10])}')
  print(f'Recall: Mean - {Average(recall[:10])} | Std - {sum([((x - Average(recall[:10])) ** 2) for x in recall[:10]]) / len(recall[:10])}')
  print(f'F1: Mean - {Average(f1[:10])} | Std - {sum([((x - Average(f1[:10])) ** 2) for x in f1[:10]]) / len(f1[:10])}')
  print(f'Accuracy: Mean - {Average(accuracy[:10])} | Std - {sum([((x - Average(accuracy[:10])) ** 2) for x in accuracy[:10]]) / len(accuracy[:10])}')
  print(f'MCC: Mean - {Average(mcc[:10])} | Std - {sum([((x - Average(mcc[:10])) ** 2) for x in mcc[:10]]) / len(mcc[:10])}')
  print('______________________________________')
  print('20 bootstraps')
  print(f'Precision: Mean - {Average(precision)} | Std - {sum([((x - Average(precision)) ** 2) for x in precision]) / len(precision)}')
  print(f'Recall: Mean - {Average(recall)} | Std - {sum([((x - Average(recall)) ** 2) for x in recall]) / len(recall)}')
  print(f'F1: Mean - {Average(f1)} | Std - {sum([((x - Average(f1)) ** 2) for x in f1]) / len(f1)}')
  print(f'Accuracy: Mean - {Average(accuracy)} | Std - {sum([((x - Average(accuracy)) ** 2) for x in accuracy]) / len(accuracy)}')
  print(f'MCC: Mean - {Average(mcc)} | Std - {sum([((x - Average(mcc)) ** 2) for x in mcc]) / len(mcc)}')


