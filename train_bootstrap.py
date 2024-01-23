import os
import time
import copy
import random
import argparse
import torch

from src.protbert import Baseline, MLP
from src.baseline_dataset import CovAbDabDataset, SMOTEDataset, MyDataset
from src.metrics import MCC
from src.data_loading_split import stratified_split
from src.training_eval import train_model, eval_model

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


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
  argparser.add_argument('-s', '--save', type=bool, help='Save checkpoints', default= True)
  argparser.add_argument('-sub', '--subset', type=int, help='Subset to train the model with', default = 0)
  argparser.add_argument('-tot', '--total', type=bool, help='Complete dataset', default = True)
  argparser.add_argument('-rep', '--repetition', type=bool, help='Repeat the non-protein class', default= False)
  argparser.add_argument('-aug', '--augmentation', type=bool, help='Augmentation of the non-protein class', default= True)
  argparser.add_argument('-smote', '--smote', type=bool, help='SMOTE augmentation', default= False)
  argparser.add_argument('-hal', '--hallucination', type=bool, help='FvHallucinator augmentation', default= False)
  argparser.add_argument('-esm', '--esm', type=bool, help='ESM augmentation', default= True)
  argparser.add_argument('-d', '--double', type=bool, help='Double dataset', default= True)
  argparser.add_argument('-b', '--bootstrap', type=bool, help='Bootstrap evaluation', default= True)
  argparser.add_argument('-nb', '--number_bootstrap', type=int, help='Number of bootstraps', default= 20)

  # Evaluation metrics
  precision = []
  recall = []
  f1 = []
  accuracy = []
  mcc = []
  args = argparser.parse_args()

  # Check for bootstrap evaluation flag
  if args.bootstrap:
    num_b = args.number_bootstrap
  else:
    num_b = 1


  for i in range(num_b):

    print(f"Subset {i}")

    # Parse arguments
    
    if args.ensemble:
      args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'norep', str(args.subset)])
    else:
      if args.total:
        if args.hallucination:
          args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'bootstrap', 'hallucination'])
        elif args.esm:
          args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'bootstrap', 'esm2'])
        elif args.repetition:
          args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'bootstrap', 'oversampling'])
        elif args.smote:
          args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'smote', 'oversampling'])
        else:
          args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'sabdab', 'old_split', 'smote', 'imbalanced'])
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
      device=device,
      save = args.save,
      smote = args.smote,
      ensemble = args.ensemble,
      model_name = args.model,
      save_name = args.save_name,
      subset = i,
      epoch_number = args.epoch_number
    )

    print("\n ## Training DONE ")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    pred, org = eval_model(
    model,
    dataloaders,
    device,
    args.smote
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


