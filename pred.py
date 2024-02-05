import os
import time
import copy
import random
import argparse
import torch

from src.protbert import Baseline, MLP
from src.baseline_dataset import CovAbDabDataset, SMOTEDataset, MyDataset
from src.metrics import MCC
from src.data_loading_split import load_data
from src.training_eval import eval_model, model_initializer

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
  argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'antiberty')
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
  argparser.add_argument('-smote', '--smote', type=bool, help='SMOTE augmentation', default= False)
  argparser.add_argument('-hal', '--hallucination', type=bool, help='FvHallucinator augmentation', default= False)
  argparser.add_argument('-esm', '--esm', type=bool, help='ESM augmentation', default= True)
  argparser.add_argument('-d', '--double', type=bool, help='Double dataset', default= False)
  argparser.add_argument('-b', '--bootstrap', type=bool, help='Bootstrap evaluation', default= False)
  argparser.add_argument('-nb', '--number_bootstrap', type=int, help='Number of bootstraps', default= 20)

  # Evaluation metrics
  precision = []
  recall = []
  f1 = []
  accuracy = []
  mcc = []
  args = argparser.parse_args()


  # Create the dataset object
  if args.smote:
    if args.model == 'antiberty':
      dataset = SMOTEDataset('/disk1/abtarget/dataset/gono/antiberty_embeddings_SMOTE_gono.csv')
    elif args.model == 'protbert':
      dataset = SMOTEDataset('/disk1/abtarget/dataset/gono/protbert_embeddings_SMOTE_gono.csv')
    else:
      exit()
  else:
    dataset = CovAbDabDataset('/disk1/abtarget/dataset/gono/Gono_test_ammino.csv')
  

  if args.threads:
    torch.set_num_threads(args.threads)

  # Train test split 
  nn_train = 0.8
  save_path = os.path.join(args.input, 'checkpoints') 

  test_data = load_data(dataset, dataset.labels, 'test')
    
  test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=1, num_workers=4, pin_memory=True
  )

  dataloaders = {
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

  
  if args.total:
    if args.hallucination:
      if args.model == 'antiberty':
        checkpoint_path = '/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_hallucination_13'
      else:
        checkpoint_path = '/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_hallucination_13' 
    elif args.esm:
      if args.model == 'antiberty':
        checkpoint_path = '/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_esm1b_2'
      else:
        checkpoint_path = '/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_esm1b_2'
    elif args.repetition:
      if args.model == 'antiberty':
        checkpoint_path = '/disk1/abtarget/checkpoints_old/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_rep'
      else:
        checkpoint_path = '/disk1/abtarget/checkpoints_old/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_rep_aug'
    elif args.smote:
      if args.model == 'antiberty':
        checkpoint_path = '/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_SMOTE'
      else:
        checkpoint_path = '/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_SMOTE'
    else:
      if args.model == 'antiberty':
        checkpoint_path = '/disk1/abtarget/checkpoints_old/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot'
        #checkpoint_path = '/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_lrscheduler'
      else:
        checkpoint_path = '/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot'
        #checkpoint_path = '/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_lrscheduler'
  else:
    if args.model == 'antiberty':
        checkpoint_path = '/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep'
    else:
        checkpoint_path = '/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep'



  model = model_initializer(checkpoint_path, model) 

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

  print('Precision: ', precision)
  print('Recall: ', recall)
  print('F1: ', f1)
  print('Accuracy: ', accuracy)
  print('MCC: ', mcc)
