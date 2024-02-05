"""
Dataset for baseline model

x = [vh_for_bert_encoding , vl_for_bert_encoding]
"""
import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from altro import fasta


class CovAbDabDataset(Dataset):
    def __init__(self, path, target=None, alignment=None):

        self._path = path

        # Load the input csv
        self._data = pd.read_csv(self._path, sep=",")
        self._len = len(self._data.index)

        # Extract the labels
        self._labels = self._data["label"].astype(float)
        #print(self._labels)

        #Extract the names
        self._names = self._data["name"]

    def __len__(self):
        return self._len

    @property
    def labels(self):
        return self._labels
    
    @property
    def name(self):
        return self._names

    def __getitem__(self, idx):

        row = self._data.iloc[idx]
        return {
            "name": row["name"],
            "VH": row["VH"],
            "VL": row["VL"],
            #"target": self._align[row["organism"]],
            "label": row["label"],
            "target": row["target"],
        }
    



class VHVLDataset(Dataset):
    def __init__(self, path, target=None, alignment=None):

        self._path = path

        # Load the input csv
        self._data = pd.read_csv(self._path, sep=",")
        self._len = len(self._data.index)

        # Extract the labels
        self._labels = self._data["label"].astype(float)
        print(self._labels)

    def __len__(self):
        return self._len

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, idx):

        row = self._data.iloc[idx]
        return {
            "name": row["name"],
            "VHVL": row["VH"] + row["VL"],
            #"VH": row["VH"],
            #"VL": row["VL"],
            #"target": self._align[row["organism"]],
            "label": row["label"],
            "target": row["target"],
        }
    
class SMOTEDataset(Dataset):
    def __init__(self, path, target=None, alignment=None):

        self._path = path

        # Load the input csv
        self._data = pd.read_csv(self._path, sep=",")
        self._len = len(self._data.index)

        # Extract the labels
        self._labels = self._data["y"].astype(float)
        print(self._labels)

    def __len__(self):
        return self._len

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, idx):

        row = self._data.iloc[idx]
        list_of_strings =[float(el) for el in row["X"].strip('][').split(', ')]
        return {
            #"X": torch.tensor(row["X"].strip('][').split(', ')),
            "X": torch.tensor(list_of_strings),
            "label": torch.tensor(row["y"]),
        }
    

class MyDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'X': torch.tensor(self.embeddings[idx]),
            'label': torch.tensor(self.labels[idx])
        }
        return sample



#if __name__ == "__main__":
#
#    dataset = MabMultiStrainBinding(
#        Path("/disk1/abtarget/mAb_dataset/dataset.csv"), None
#    )
#     print("# Dataset created")