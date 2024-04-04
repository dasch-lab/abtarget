import random
import argparse
import torch

from src.protbert import Baseline
from src.baseline_dataset import SAbDabDataset
from src.data_loading_split import load_data
from src.training_eval import model_initializer, eval_model1_model2, final_score_eval

def ensemble_model_initializer(path, name):
  model = Baseline(args.batch_size, device, args.n_class, freeze_bert=True, model_name = name) 
  model = model_initializer(path, model)
  return model


if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser('Baseline for Abtarget classification', add_help=False) #, fromfile_prefix_chars="@")
  argparser.add_argument('-i', '--input', help='input model folder', type=str, default = "/disk1/abtarget/dataset")
  argparser.add_argument('-ch', '--checkpoint', help='checkpoint folder', type=str, default = "/disk1/abtarget")
  argparser.add_argument('-t', '--threads',  help='number of cpu threads', type=int, default=None)
  argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'protbert')
  argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=200)
  argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=1)
  argparser.add_argument('-r', '--random', type=int, help='Random seed', default=None)
  argparser.add_argument('-c', '--n_class', type=int, help='Number of classes', default=2)
  argparser.add_argument('-o', '--optimizer', type=str, help='Optimizer: SGD or Adam', default='Adam')
  argparser.add_argument('-l', '--lr', type=float, help='Learning rate', default=3e-5)
  argparser.add_argument('-cr', '--criterion', type=str, help='Criterion: BCE or Crossentropy', default='Crossentropy')
  argparser.add_argument('-ens', '--ensemble', type=bool, help='Ensable models', default=False)
  argparser.add_argument('-sin', '--one_encoder', type=bool, help='One encoder (ProtBERT or AntiBERTy)', default=False)


  # Parse arguments
  args = argparser.parse_args()
  
  # Load Data
  dataset = SAbDabDataset('/disk1/abtarget/dataset/sabdab/split/sabdab_200423_test_norep.csv')
  

  if args.threads:
    torch.set_num_threads(args.threads)
  
  test_data = load_data(dataset, dataset.labels, name_sub = 'test')
  
  # Data Loader
  test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True
  )

  # Set the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Using device {0}'.format(device))

  # Train script
  dataloaders = {
    "test": test_loader
  }

  # Select model
  if args.ensemble:
    model1a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/0/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_0_best_accuracy', model_name = 'antiberty')
    model2a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/1/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_1_best_accuracy', model_name = 'antiberty')
    model3a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/2/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_2_best_accuracy', model_name = 'antiberty' )
    model4a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/3/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_3_best_accuracy', model_name = 'antiberty' )
    model5a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/4/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_4_best_accuracy', model_name = 'antiberty' )
    model6a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/5/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_5_best_accuracy', model_name = 'antiberty' )
    model7a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/6/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_6_best_accuracy', model_name = 'antiberty' )
    model8a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/7/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_7_best_accuracy', model_name = 'antiberty' )
    model9a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/8/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_8_best_accuracy', model_name = 'antiberty' )
    model10a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/10/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_10_best_accuracy', model_name = 'antiberty' )
    model11a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/11/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_11_best_accuracy', model_name = 'antiberty' )
    model12a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/12/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_12_best_accuracy', model_name = 'antiberty' )
    model13a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/13/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_13_best_accuracy', model_name = 'antiberty' )
    model14a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/14/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_14_best_accuracy', model_name = 'antiberty' )
    model15a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/15/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_15_best_accuracy', model_name = 'antiberty' )
    model16a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/16/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_16_best_accuracy', model_name = 'antiberty' )
    model17a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/17/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_17_best_accuracy', model_name = 'antiberty' )
    model18a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/18/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_18_best_accuracy', model_name = 'antiberty' )
    model19a = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/antiberty/ensemble/19/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_19_best_accuracy', model_name = 'antiberty' )
    model1p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/0/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_0_best_accuracy', model_name = 'protbert' )
    model2p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/1/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_1_best_accuracy', model_name = 'protbert' )
    model3p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/2/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_2_best_accuracy', model_name = 'protbert' )
    model4p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/3/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_3_best_accuracy', model_name = 'protbert' )
    model5p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/4/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_4_best_accuracy', model_name = 'protbert' )
    model6p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/5/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_5_best_accuracy', model_name = 'protbert' )
    model7p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/6/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_6_best_accuracy', model_name = 'protbert' )
    model8p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/7/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_7_best_accuracy', model_name = 'protbert' )
    model9p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/8/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_8_best_accuracy', model_name = 'protbert' )
    model10p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/10/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_10_best_accuracy', model_name = 'protbert' )
    model11p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/11/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_11_best_accuracy', model_name = 'protbert' )
    model12p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/12/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_12_best_accuracy', model_name = 'protbert' )
    model13p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/13/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_13_best_accuracy', model_name = 'protbert' )
    model14p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/14/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_14_best_accuracy', model_name = 'protbert' )
    model15p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/15/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_15_best_accuracy', model_name = 'protbert' )
    model16p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/16/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_16_best_accuracy', model_name = 'protbert' )
    model17p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/17/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_17_best_accuracy', model_name = 'protbert' )
    model18p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/18/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_18_best_accuracy', model_name = 'protbert' )
    model19p = ensemble_model_initializer('/disk1/abtarget/checkpoints_old/protbert/ensemble/19/protbert_50_16_Adam_Crossentropy_True_sabdab_new_split_norep_19_best_accuracy', model_name = 'protbert' )
    list_model1=[model1a, model2a, model3a, model4a, model5a, model6a, model7a, model8a, model9a, model10a, model11a, model12a, model13a, model14a, model15a, model16a, model17a, model18a, model19a]
    list_model2=[model1p, model2p, model3p, model4p, model5p, model6p, model7p, model8p, model9p, model10p, model11p, model12p, model13p, model14p, model15p, model16p, model17p, model18p, model19p]
  else:
    model1 = Baseline(args.batch_size, device, args.n_class, freeze_bert=True, model_name = 'protbert') 
    model2 = Baseline(args.batch_size, device, args.n_class, freeze_bert=True, model_name = 'antiberty')
    #model1 = model_initializer('/disk1/abtarget/checkpoints_old/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_rep_aug', model1)
    #model2 = model_initializer('/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_rep_aug', model2)
    model1 = model_initializer('/disk1/abtarget/checkpoints/protbert/single/protbert_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_esm1b_2', model1)
    model2 = model_initializer('/disk1/abtarget/checkpoints/antiberty/single/antiberty_50_16_Adam_Crossentropy_True_sabdab_old_split_norep_tot_esm1b_2', model2)
    list_model1 = [model1]
    list_model2 = [model2]

  # Train model
  org, pred = eval_model1_model2(
    dataloaders = dataloaders,
    device = device,
    single = args.one_encoder,
    list_model1 = list_model1,
    list_model2 = list_model2
  )

  final_score_eval(pred, org)
