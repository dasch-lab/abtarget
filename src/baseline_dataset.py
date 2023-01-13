"""
Dataset for baseline model

x = [vh_for_bert_encoding , vl_for_bert_encoding]
"""


from pathlib import Path
import torch
import pandas as pd
from sklearn import preprocessing

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


class MabDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()

        csv = pd.read_csv(path, sep=",")
        csv = csv[csv["Ab or Nb"] == "Ab"]
        csv = csv[csv["VHorVHH"] != "ND"]
        csv = csv[csv["VL"] != "ND"]
        self.csv = csv.reset_index(drop=True)

    def __len__(self):
        # Returns length
        return len(self.csv)

    # TODO which is the binary task ??
    def __getitem__(self, idx):

        xxx = self.csv.iloc[idx][["VHorVHH", "VL"]].tolist()
        xxx = [" ".join(list(_)) for _ in xxx]
        label = None
        pass
        return xxx, label


class MabBinaryStrainBinding(torch.utils.data.Dataset):
    def __init__(self, path, variant="SARS-CoV2_WT"):
        super().__init__()

        csv = pd.read_csv(path, sep="\t")
        # TODO filter for organism
        if variant:
            csv = csv[csv.organism == variant].reset_index(drop=True)
        self.csv = csv
        self.variant = variant

    def __len__(self):
        # Returns length
        return len(self.csv)

    # Task is binding prediction
    def __getitem__(self, idx):

        vh, vl, label = self.csv.iloc[idx][["VH", "VL", "bind"]].tolist()
        xxx = [vh, vl]
        xxx = [" ".join(list(_)) for _ in xxx]
        return xxx, label


class MabMultiStrainBinding(torch.utils.data.Dataset):
    def __init__(self, path, variant=None):
        super().__init__()

        csv = pd.read_csv(path, sep="\t")
        # filter for organism
        if variant:
            csv = csv[csv.organism == variant].reset_index(drop=True)
            self.variant = variant

        # Get the variants as multilabel classification
        FILTER_4_VARIANTS = 800
        var_list = csv.organism.value_counts()[
            csv.organism.value_counts() >= FILTER_4_VARIANTS
        ].index.tolist()

        df = pd.DataFrame(columns=["name", "VH", "VL", "target"] + var_list)
        for ii, mab in enumerate(csv.name.unique()):
            tmp_df = csv[csv.name == mab]
            mask = tmp_df.organism.isin(var_list)
            tmp_df_filtered = tmp_df[mask]
            if tmp_df_filtered.values.tolist():
                # print(ii)
                # import pdb ; pdb.set_trace()
                df1 = tmp_df_filtered[["organism", "bind"]].T.reset_index(drop=True)
                df1.rename(columns=df1.iloc[0], inplace=True)
                df1.drop(df1.index[0], inplace=True)
                df1.insert(0, "target", tmp_df_filtered.iloc[0].target)
                df1.insert(0, "VL", tmp_df_filtered.iloc[0].VL)
                df1.insert(0, "VH", tmp_df_filtered.iloc[0].VH)
                df1.insert(0, "name", mab)
                df1.index = pd.Index([ii])
                df1 = df1.loc[
                    :, ~df1.columns.duplicated()
                ].copy()  # drop duplicated columns
            else:
                # print(ii,'-- binds no variant')
                data = tmp_df[["name", "VH", "VL", "target"]].values.tolist()[0] + [
                    -1
                ] * len(var_list)
                df1 = pd.DataFrame(
                    [data], columns=["name", "VH", "VL", "target"] + var_list
                )

            df = pd.concat([df, df1])

        self.csv = df.fillna(-1)  # not-bind/bind 0/1 ; -1 not-tested

        # TODO binarize the labels
        self.csv[var_list] = self.csv[var_list].clip(0)

        self.var_list = var_list

    def __len__(self):
        # Returns length
        return len(self.csv)

    # # Task is binding prediction
    # def __getitem__(self, idx):

    #     vh, vl = self.csv.iloc[idx][["VH", "VL"]].tolist()
    #     label = torch.tensor(self.csv.iloc[idx][self.var_list].tolist())
    #     xxx = [vh, vl]
    #     xxx = [" ".join(list(_)) for _ in xxx]
    #     return xxx, label

    # Task is binding prediction
    def __getitem__(self, idx):

        xxx = self.csv.iloc[idx][["VH", "VL"]].tolist()
        label = torch.tensor(self.csv.iloc[idx][self.var_list].tolist())
        xxx = [" ".join(list(_)) for _ in xxx]
        return xxx, label

class MabMultiStrainBinding(torch.utils.data.Dataset):
    def __init__(self, path, variant=None):
        super().__init__()

        csv = pd.read_csv(path, sep=",")

        df = pd.DataFrame(columns=["name", "VH", "VL", "target"])
        for ii, mab in enumerate(csv.name.unique()):

            tmp_df = csv[csv.name == mab]
            data = tmp_df[["name", "VH", "VL", "target"]].values.tolist()[0]
            df1 = pd.DataFrame([data], columns=["name", "VH", "VL", "target"])
            df = pd.concat([df, df1], ignore_index=True)

        lb = preprocessing.LabelBinarizer()
        df["target"] = lb.fit_transform(df["target"].tolist())

    def __len__(self):
        # Returns length
        return len(self.csv)

    # Task is binding prediction
    def __getitem__(self, idx):

        xxx = self.csv.iloc[idx][["VH", "VL"]].tolist()
        label = torch.tensor(self.csv.iloc[idx]["target"].tolist())
        xxx = [" ".join(list(_)) for _ in xxx]
        return xxx, label

import os
import pandas as pd

from torch.utils.data import Dataset
# from src import configs
from src import fasta


class CovAbDabDataset(Dataset):
    def __init__(self, path, target=None, alignment=None):

        self._path = path

        # Load the input csv
        self._data = pd.read_csv(self._path, sep=",")
        self._len = len(self._data.index)

        # Extract the labels
        self._labels = self._data["label"]
        print(self._labels)

    def __len__(self):
        return self._len

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, idx):

        row = self._data.iloc[idx]
        return {
            "VH": row["VH"],
            "VL": row["VL"],
            #"target": self._align[row["organism"]],
            "label": row["label"],
        }

    """
     @staticmethod
    def parseAlignment(name=None):

        # This function loads the alignment file, extract the annotation and masks unwanted residues

        # Get paths from config
        align_path = os.path.join(
            configs.get("root", ""), configs.get("alignment", None)
        )
        bed_path = os.path.join(
            configs.get("root", ""), configs.get("annotation", None)
        )
        if not os.path.exists(align_path) or not os.path.exists(bed_path):
            raise Exception("Unable to find alignment or annotation file")

        # Parse alignment fasta
        # The fasta header is composed by <name>;<accession>
        result = {}
        for (name, sequence) in fasta.parse(align_path):
            name = name.split(";")[0]
            result[name] = sequence

        # Parse bed file
        # The bed file needs to have the first 4 columns:
        # chr|start|end|name
        annotation = {}
        with open(bed_path, "r") as handle:
            for line in handle:
                line = line.strip().split()
                if len(line) < 4:
                    raise Exception("Invalid bed file format")

                name = line[3]
                annotation[name] = {"start": int(line[1]), "end": int(line[2])}

        # Do we need to extract the sequence?
        if not name:
            return result

        # Extract only part of the sequence
        fragment = annotation[name] if name in annotation else None
        if not fragment:
            raise Exception("Invalid annotation name: '{0}'".format(name))

        # Apply annotation
        for key in result:
            value = result[key][fragment["start"] : fragment["end"]]
            result[key] = value

        # Do we need to mask?
        mask = configs.get("mask", [])
        if len(mask) == 0:
            return result

        # Mask unwanted residues
        for key in result:
            value = "".join([result[key][i] for i in mask])
            result[key] = value

        return result 
        
    """


"""
# generate the dataset.txt in the <./test> folder for the training
python dataset.py -i ./data/covabdab_260722.csv -o ./test

"""


#if __name__ == "__main__":
#
#    dataset = MabMultiStrainBinding(
#        Path("/disk1/abtarget/mAb_dataset/dataset.csv"), None
#    )
#     print("# Dataset created")


"""
# Organism-list:
SARS-CoV2_WT                  8309
SARS-CoV2_Omicron-BA1         1452
SARS-CoV2_Omicron-BA2         1003
SARS-CoV2_Delta                873
SARS-CoV2_Omicron-BA1.1        752
SARS-CoV2_Beta                 723
SARS-CoV2_Omicron-BA2.12.1     696
SARS-CoV2_Omicron-BA3          664
SARS-CoV2_Omicron-BA2.13       662
SARS-CoV2_Gamma                497
SARS-CoV2_Alpha                320
SARS-CoV2_Epsilon               72
SARS-CoV2_Mu                    22
SARS-CoV2_Kappa                 17
SARS-CoV2_Eta                   15
SARS-CoV2_Iota                  14
SARS-CoV2_Lambda                14
SARS-CoV2_Omicron-BA2.75         8
SARS-CoV2_Omicron-BA2.11         7
SARS-CoV2_Omicron-BA2.4          7
SARS-CoV2_Omicron-BA2.5          7
SARS-CoV2_Omicron-BA2.38         4
SARS-CoV2_Omicron                3
SARS-CoV2_Omicron-BA4            1

"""

"""
## TESTING

path = Path("/data2/dcardamone/deepcov/data/covabdab_260722.csv")
dataset = MabBinaryStrainBinding(path_dataset_txt,None)

path_dataset_txt = Path('/data2/dcardamone/deepcov/test/dataset.txt')
dataset = MabMultiStrainBinding(path_dataset_txt,None)


"""