import os
import random
import argparse
import torch

from src.protbert import MLP
from src.baseline_dataset import SMOTEDataset
from src.data_loading_split import load_data
from src.training_eval import model_initializer, final_score_eval, eval_model1_model2_smote


if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser('Baseline for Abtarget classification', add_help=False) #, fromfile_prefix_chars="@")
  argparser.add_argument('-i', '--input', help='input model folder', type=str, default = "/disk1/abtarget/dataset")
  argparser.add_argument('-ch', '--checkpoint', help='checkpoint folder', type=str, default = "/disk1/abtarget")
  argparser.add_argument('-t', '--threads',  help='number of cpu threads', type=int, default=None)
  argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'antiberty')
  argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=200)
  argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=1)
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
  dataset1 = SMOTEDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep_protbert_embeddings.csv')
  dataset2 = SMOTEDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep_antiberty_embeddings.csv')
  
  

  if args.threads:
    torch.set_num_threads(args.threads)

  # Train test split 
  nn_train = 0.8
  save_path = os.path.join(args.input, 'checkpoints') 
  test_data_protbert = load_data(dataset1, dataset1.labels, name_sub = 'test')
  test_data_antiberty = load_data(dataset2, dataset2.labels, name_sub = 'test')
  print('Done')

    
  test_loader_protbert = torch.utils.data.DataLoader(
    test_data_protbert, batch_size=args.batch_size, num_workers=4, pin_memory=True
  )

  test_loader_antiberty = torch.utils.data.DataLoader(
    test_data_antiberty, batch_size=args.batch_size, num_workers=4, pin_memory=True
  )

  # Set the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device {0}'.format(device))

  dataloader1 = {
    "test": test_loader_protbert
  }

  dataloader2 = {
    "test": test_loader_antiberty
  }

  
  model1 = MLP(args.batch_size, device, nn_classes=args.n_class, model_name='protbert')
  model2 = MLP(args.batch_size, device, nn_classes=args.n_class, model_name = 'antiberty')
  model1 = model_initializer('/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_SMOTE', model1)
  model2 = model_initializer('/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_SMOTE', model2)

  # Train model
  org, pred = eval_model1_model2_smote(
    model1,
    model2,
    dataloader1,
    dataloader2,
    device
  )

  final_score_eval(pred, org)