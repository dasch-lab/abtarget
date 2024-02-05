import torch
import random

def load_data(dataset : torch.utils.data.Dataset, labels, name_sub):

  '''
  Load the dataset 
  '''

  # Get classes
  classList = list(set(labels))
  resultList = {
    name_sub: []
  }

  classData = {}
  for name in classList:

    # Get subsample of indexes for this class
    classData[name] = [ idx for idx, label in enumerate(labels) if label == name ]

  classStats = {
    name_sub: {}
  }
  for name in classList:
    datasetList = classData[name]
    

    # Update stats
    classStats[name_sub][name] = len(datasetList)
    # Concatenate indexes
    resultList[name_sub].extend(datasetList)

  dataset_data = torch.utils.data.Subset(dataset, resultList[name_sub])

  return dataset_data


def stratified_split(dataset : torch.utils.data.Dataset, labels, step, hallucination = False, esm = False, double = False, names = None, subset_size = 50, rep = False, tot = False):

  '''
  Split the dataset proportionally according to the sample label
  '''

  print('here')

  # Get classes
  classList = list(set(labels))
  resultList = {
    'test': [],
    'val':[],
    'train': []
  }

  classData = {}
  for name in classList:
    # Get subsample of indexes for this class
    classData[name] = [idx for idx, label in enumerate(labels) if label == name]

  print('here')

  classStats = {
    'train':{},
    'val': {},
    'test': {}
  }

  random.seed(step*4)
  max_len = max(len(classData[0]) - subset_size*2, len(classData[1]) - subset_size*2)
  min_len = min(len(classData[0]) - subset_size*2, len(classData[1]) - subset_size*2)

  for name in classList:
    if name == 1 and (hallucination or esm):
      trainList, valList, testList = sample_hallucination_esm(names, classData[name], subset_size)
      if hallucination:
          trainList = random.sample(trainList, min_len)
      elif double:
          trainList = random.sample(trainList*2, max_len)
    else:
      testList = random.sample(classData[name], subset_size)
      train_val_List = [ idx for idx in classData[name] if idx not in testList]
      valList = random.sample(train_val_List, subset_size)

      if tot:
        trainList = [idx for idx in train_val_List if idx not in valList]
        if rep and name == 1:
          repl = (len(classData[0]) - subset_size*2) // (len(classData[1]) - subset_size*2) +1
          trainList = random.sample(trainList*repl, max_len)
      else:
        if name == 0:
          trainList = random.sample([idx for idx in train_val_List if idx not in valList], min_len)
        else:
          trainList = [idx for idx in train_val_List if idx not in valList]

    print(len(trainList))
    print(len(valList))
    print(len(testList))

    # Update stats
    classStats['train'][name] = len(trainList)
    classStats['val'][name] = len(valList)
    classStats['test'][name] = len(testList)

    # Concatenate indexes
    resultList['train'].extend(trainList)
    resultList['val'].extend(valList)
    resultList['test'].extend(testList)

  # Construct the test and train datasets
  train_data = torch.utils.data.Subset(dataset, resultList['train'])
  val_data = torch.utils.data.Subset(dataset, resultList['val'])
  test_data = torch.utils.data.Subset(dataset, resultList['test'])

  return train_data, val_data, test_data

def sample_hallucination_esm(names, idx_list, subset_size):
  id_name_dict = {}
  names = [names[idx].lower() for idx in idx_list]
  for id, name in zip(idx_list, names):
    if name in id_name_dict:
      id_name_dict[name].append(id)
    else:
      id_name_dict[name] = [id]

  sample_names = random.sample(list(id_name_dict.keys()), subset_size*2)
  test_list = sample_without_aug(sample_names[:subset_size], id_name_dict)
  val_list = sample_without_aug(sample_names[subset_size:], id_name_dict)
  train_list = []
  train_list.extend(value for values in id_name_dict.values() for value in values)

  return train_list, test_list, val_list

def sample_without_aug(sample_names, id_name_dict):
  list_set = []
  for name in sample_names:
    idx = id_name_dict[name]
    list_set.append(idx[0])
    id_name_dict.pop(name, None)
  return list_set