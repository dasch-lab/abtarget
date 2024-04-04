import random
import argparse
import torch
import numpy as np

from src.protbert import Baseline
from src.baseline_dataset import SAbDabDataset
from src.training_eval import model_initializer, train_model, eval_model1_model2, final_score_eval
import warnings
warnings.filterwarnings('ignore')

def ensemble_model_initializer(path, name):
  model = Baseline(args.batch_size, device, args.n_class, freeze_bert=True, model_name = name) 
  model = model_initializer(path, model)
  return model

def controlled_split(dataset : torch.utils.data.Dataset, labels, subset = 0, bootstrap = 0, sample_size = 50):

  '''
  Split the dataset proportionally according to the sample label
  '''

  # Get classes
  classList = list(set(labels))
  resultList = {
    'test': [],
    'val':[],
    'train': []
  }

  random.seed(bootstrap)
  classData = {}
  for name in classList:
    # Get subsample of indexes for this class
    classData[name] = [ idx for idx, label in enumerate(labels) if label == name ]

  # Get shorter element
  shorter_class = min(classData.items(), key=lambda x: len(x[1]))[0]

  classStats = {
    'train': {},
    'val': {},
    'test': {}
  }

  for name in classList:
    testList = random.sample(classData[name], sample_size)
    train_val_List = [ idx for idx in classData[name] if idx not in testList]
    valList = random.sample(train_val_List, sample_size)
    trainList = [idx for idx in train_val_List if idx not in valList]

    if name != shorter_class:
      classData[name] = [ idx for idx, label in enumerate(labels) if label == name and label in trainList ]
      #train_size = len(classData[shorter_class]) - 2*sample_size
      #step = int(2 * train_size/3)

      #classDatasubset = []
      #for base in range(0,len(trainList),step):
      #  if base+train_size > len(trainList):
      #    classDatasubset.append(trainList[base:])
      #    break
      #  else:
      #    classDatasubset.append(trainList[base:base+train_size])

      
      class_0_samples = [idx for idx in trainList if idx in classData[name]]
      random.shuffle(trainList)
      min_class = 215
      trainList = []
      i = 0
      for base in range(0,len(class_0_samples),int(2 * min_class/3)):
        print(i+1)
        i = i+1
        subset_start = base
        subset_end = base + min_class
        if subset_end < len(class_0_samples):
          trainList.append(class_0_samples[subset_start:subset_end])
        else:
          break
      
      


      #trainList =  classDatasubset[subset]
      

    # Update stats
    classStats['train'][name] = len(trainList)
    classStats['val'][name] = len(valList)
    classStats['test'][name] = len(testList)

    # Concatenate indexes
    resultList['train'].append(trainList)
    resultList['val'].extend(valList)
    resultList['test'].extend(testList)

  # Shuffle index lists
  #for key in resultList:
    #random.shuffle(resultList[key])
    #print('{0} dataset:'.format(key))
    #for name in classList:
    #  print(' Class {0}: {1}'.format(name, classStats[key][name]))
  
  # Construct the test and train datasets



  return resultList


if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser('Baseline for Abtarget classification', add_help=False) #, fromfile_prefix_chars="@")
  argparser.add_argument('-i', '--input', help='input model folder', type=str, default = "/disk1/abtarget/dataset")
  argparser.add_argument('-ch', '--checkpoint', help='checkpoint folder', type=str, default = "/disk1/abtarget/checkpoints/ensemble")
  argparser.add_argument('-t', '--threads',  help='number of cpu threads', type=int, default=None)
  argparser.add_argument('-m', '--model', type=str, help='Which model to use: protbert, antiberty, antiberta', default = 'protbert')
  argparser.add_argument('-t1', '--epoch_number', help='training epochs', type=int, default=50)
  argparser.add_argument('-t2', '--batch_size', help='batch size', type=int, default=1)
  argparser.add_argument('-r', '--random', type=int, help='Random seed', default=None)
  argparser.add_argument('-c', '--n_class', type=int, help='Number of classes', default=2)
  argparser.add_argument('-o', '--optimizer', type=str, help='Optimizer: SGD or Adam', default='Adam')
  argparser.add_argument('-l', '--lr', type=float, help='Learning rate', default=3e-5)
  argparser.add_argument('-cr', '--criterion', type=str, help='Criterion: BCE or Crossentropy', default='Crossentropy')
  argparser.add_argument('-ens', '--ensemble', type=bool, help='Ensable models', default=True)
  argparser.add_argument('-tr', '--pretrain', type=bool, help='Freeze encoder', default= True)
  argparser.add_argument('-sin', '--one_encoder', type=bool, help='One encoder (ProtBERT or AntiBERTy)', default=False)
  argparser.add_argument('-data', '--dataset', type=str, help='Dataset', default= "/disk1/abtarget/dataset/sabdab/split/sabdab_200423_norep.csv")

  args = argparser.parse_args()
  dataset = SAbDabDataset(args.dataset)
      
  resultList = controlled_split(dataset, dataset.labels, subset = 0, bootstrap = 0, sample_size = 50)
  val_data = torch.utils.data.Subset(dataset, resultList['val'])
  test_data = torch.utils.data.Subset(dataset, resultList['test'])
  
  precision = []
  recall = []
  f1 = []
  accuracy = []
  mcc = []

  for j in range(19):
    print('Bootstrap: ', j)

    for i in range(19):

      print('Ensable: ', i)

      subset = i
      idx = []
      idx.extend(resultList['train'][1])
      idx.extend(resultList['train'][0][subset])
      train_data = torch.utils.data.Subset(dataset, idx)

      save_name_1 = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'bootstrap'])

      args.save_name = '_'.join([args.model, str(args.epoch_number), str(args.batch_size), args.optimizer, args.criterion, str(args.pretrain), 'bootstrap', str(subset)])

      print(f"Model: {args.model} | Epochs: {args.epoch_number} | Batch: {args.batch_size} | Optimizer: {args.optimizer} | Criterion: {args.criterion} | Learning rate: {args.lr}")
      

      if args.threads:
        torch.set_num_threads(args.threads)
      
      # Data Loader
      train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
      )
      val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True
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

      model = None
      model_name = args.model.lower()

      model = Baseline(args.batch_size, device, nn_classes=args.n_class, freeze_bert=args.pretrain, model_name=args.model) 


      if args.criterion == 'Crossentropy':
        criterion = torch.nn.CrossEntropyLoss().to(device) #TODO
      else:
        criterion = torch.nn.BCELoss().to(device)

      if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
      else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
      
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

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
          save = True,
          smote = False,
          ensemble = args.ensemble,
          model_name = args.model,
          save_name = args.save_name,
          subset = i,
          epoch_number = args.epoch_number
        )

    # Select model
    model1a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_0_best_accuracy']), args.model)
    model2a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_1_best_accuracy']), args.model)
    model3a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_2_best_accuracy']), args.model )
    model4a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_3_best_accuracy']), args.model )
    model5a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_4_best_accuracy']), args.model )
    model6a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_5_best_accuracy']), args.model )
    model7a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_6_best_accuracy']), args.model )
    model8a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_7_best_accuracy']), args.model )
    model9a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_8_best_accuracy']), args.model )
    model10a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_9_best_accuracy']), args.model )
    model11a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_10_best_accuracy']), args.model )
    model12a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_11_best_accuracy']), args.model )
    model13a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_12_best_accuracy']), args.model )
    model14a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_13_best_accuracy']), args.model )
    model15a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_14_best_accuracy']), args.model )
    model16a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_15_best_accuracy']), args.model )
    model17a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_16_best_accuracy']), args.model )
    model18a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_17_best_accuracy']), args.model )
    model19a = ensemble_model_initializer('/'.join([args.checkpoint,save_name_1 + '_18_best_accuracy']), args.model )
    list_model1=[model1a, model2a, model3a, model4a, model5a, model6a, model7a, model8a, model9a, model10a, model11a, model12a, model13a, model14a, model15a, model16a, model17a, model18a, model19a]

    # Train model
    org, pred = eval_model1_model2(
      dataloaders = dataloaders,
      device = device,
      single = True,
      list_model1 = list_model1
    )

    precision1, recall1, f1_1, accuracy1, mcc1 = final_score_eval(pred, org)
    precision.append(precision1)
    recall.append(recall1)
    f1.append(f1_1)
    accuracy.append(accuracy1)
    mcc.append(mcc)
  
  print('Precision = ', precision)
  print('Recall = ', recall)
  print('F1 = ', f1)
  print('Accuracy = ', accuracy)
  print('MCC = ', mcc)

  print('20 bootstraps')
  print(f'Precision: {round(np.mean(precision),3)} ± {round(np.std(precision),3)}')
  print(f'Recall: {round(np.mean(recall),3)} ± {round(np.std(recall),3)}')
  print(f'F1: {round(np.mean(f1),3)} ± {round(np.std(f1),3)}')
  print(f'Accuracy: {round(np.mean(accuracy),3)} ± {round(np.std(accuracy),3)}')
  print(f'MCC: {round(np.mean(mcc),3)} ± {round(np.std(mcc),3)}')
