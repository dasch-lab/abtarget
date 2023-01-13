from pandas import *
import traceback
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from antiberty import AntiBERTy, get_weights

from transformers import BertModel, BertTokenizer

from transformers import (
    RobertaTokenizer,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def csv_2_list(csv_name):
    data = read_csv(csv_name)
    filename = data['File_name'].tolist()
    vh = data['VH'].tolist()
    vl = data['VL'].tolist()
    target = data['target'].tolist()
    return filename, vh, vl, target

def protbert_embedding(sequences):

    # 1. Load the vocabulary and ProtBert Model
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert")

    # 2. Load the model into the GPU if avilabile and switch to inference mode
    model = model.to(device)
    model = model.eval()

    # 3. Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
    pre_sequence = [re.sub(r"[UZOB-]", "X", sequence) for sequence in sequences]

    # 4. Tokenize, encode sequences and load it into the GPU if possibile
    ids = tokenizer.batch_encode_plus(pre_sequence, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # 5. Extracting sequences' features and load it into the CPU if needed
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
    embedding = embedding.cpu().numpy()

    # 6. Remove padding ([PAD]) and special tokens ([CLS],[SEP]) that is added by Bert model
    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1:seq_len-1]
        features.append(seq_emd)
    
    return features

if __name__ == "__main__":
    filename, vh, vl, target = csv_2_list("/disk1/abtarget/mAb_dataset/dataset.csv")
    binary_class = [0]*(len(target)-1)
    n = target.count('Protein')
    binary_class[0:n] = [1]*n
    binary_class.append([0]*(len(target)-len(binary_class)))
    sequences = [list(el) for el in zip(vh, vl)]

    x_train, x_test, y_train, y_test = train_test_split(sequences, binary_class)

    

    #if name == 'protbert':
    #    vh_features = protbert_embedding(vh)
    #    vl_features = protbert_embedding(vl)
    #elif name == 'antiberta':
    #    # 1. initialize tokenizer
    #    TOKENIZER_DIR = "antibody-tokenizer"
    #    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_DIR, max_len=150)
    #
    #    MODEL_DIR = "antiberta-base"# We initialise a model using the weights from the pre-trained model
    #    model = RobertaForTokenClassification.from_pretrained(MODEL_DIR, num_labels=2)
    #    tokenized_input = tokenizer(vl, return_tensors='pt', padding=True)
    #    predicted_logits = model(**tokenized_input)
    #else:
    #    pass