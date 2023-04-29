import torch
from torch import nn
from transformers import BertModel, BertTokenizer, logging, RobertaTokenizer, RobertaModel, Trainer, TrainingArguments
from huggingface_hub import login
#login()

PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert_bfd"
MAX_LEN = 512
logging.set_verbosity_error()

TOKENIZER_DIR = "antibody-tokenizer"
MODEL_DIR = "antiberta-base"

from antiberty import AntiBERTy, get_weights


# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {0}".format(device))

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x
            sampled_noise = (self.noise.expand(*x.size()).float().normal_()).to(device) * scale
            x = x + sampled_noise
        return x 


class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, add_pooling_layer=False).to(device)

    def forward(self, x):

        x = [ ' '.join(list(i)) for i in x ]
        x = self.tokenizer(
            x,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        x = self.bert(**x)
        return torch.mean(x["last_hidden_state"], 1)

class AntibertaEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_DIR, max_len=MAX_LEN, use_auth_token=True, token = True)
        self.antiberta = RobertaModel.from_pretrained(MODEL_DIR, use_auth_token=True).to(device)

    def forward(self, x):

        x = [ ' '.join(list(i)) for i in x ]
        x = self.tokenizer(
            x,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)

        x = self.antiberta(**x)
        return torch.mean(x["last_hidden_state"], 1)


class AntibertyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
        self.antiberty = BertModel.from_pretrained(get_weights()).to(device)

    def forward(self, x):

        x = [ ' '.join(list(i)) for i in x ]
        x = self.tokenizer(
            x,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)

        x = self.antiberty(**x)
        return torch.mean(x["last_hidden_state"], 1)

    

class Baseline(nn.Module):
    """
    # Schema
    [BERT-VH]--|
               |--[CONCAT]--[LayerNorm,Linear, Gelu]x2--> Classification (0/1) protein or non-protein
    [BERT-VL]--|
    # Note: criterion = torch.nn.CrossEntropyLoss()
    """

    def __init__(self, batch_size, device, nn_classes=2, freeze_bert=True, model_name = 'protbert'):
        super().__init__()

        self.dropout = 0.8
        #self.embedding_length = 1024
        #self.embedding_length = MAX_LEN
        self.batch_size = batch_size
        self.device = device


        if model_name == 'protbert':
            self.encoder = BertEncoder()
            self.embedding_length = MAX_LEN*2
        elif model_name == 'antiberta':
            self.encoder = AntibertaEncoder()
            self.embedding_length = MAX_LEN
        else:
            self.encoder = AntibertyEncoder()
            self.embedding_length = MAX_LEN

        
        if freeze_bert:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        
        self.freeze_bert = freeze_bert


        # Projection for the concatenated embeddings
        
        projection = nn.Sequential(
            nn.Linear(self.embedding_length * 2, self.embedding_length),
            #nn.BatchNorm1d(self.embedding_length),
            #nn.SELU()
            nn.LeakyReLU(),
            #nn.GELU(),
            #nn.LayerNorm(self.embedding_length),
            #nn.BatchNorm1d(self.embedding_length),
            #nn.Dropout(p=0.4),
            #nn.ReLU(),
        )
    
        self.projection = projection.to(device)
        #classification_dim = min([_.out_features for _ in projection.modules() if isinstance(_, nn.Linear)])
        # assert classification_dim == 512
        classification_dim = self.embedding_length
        print(f"Classification_dim: {classification_dim}")

        # binary classification head
        head = nn.Sequential(
            nn.Linear(classification_dim, nn_classes),
        )
    

        self.head = head.to(device)

    
    def forward(self, x):

        vh = x['VH']
        vl = x['VL']
        xvh = self.encoder(vh)
        xvl = self.encoder(vl)
        x = torch.cat((xvh, xvl), 1)
        #x = torch.add(xvh, xvl)

        x = self.projection(x)
        x = self.head(x)

        return x

class BaselineOne(nn.Module):
    """
    # Schema
    [BERT-VH]--|
               |--[CONCAT]--[LayerNorm,Linear, Gelu]x2--> Classification (0/1) protein or non-protein
    [BERT-VL]--|
    # Note: criterion = torch.nn.CrossEntropyLoss()
    """

    def __init__(self, batch_size, device, nn_classes=2, freeze_bert=True, model_name = 'protbert', train_m = True):
        super().__init__()

        self.dropout = 0.8
        #self.embedding_length = 1024
        #self.embedding_length = MAX_LEN
        self.batch_size = batch_size
        self.device = device
        self.train_m = train_m 


        if model_name == 'protbert':
            self.encoder = BertEncoder()
            self.embedding_length = MAX_LEN*2
        elif model_name == 'antiberta':
            self.encoder = AntibertaEncoder()
            self.embedding_length = MAX_LEN
        else:
            self.encoder = AntibertyEncoder()
            self.embedding_length = MAX_LEN

        
        if freeze_bert:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        
        self.freeze_bert = freeze_bert
        self.noise = GaussianNoise()


        # Projection for the concatenated embeddings
        
        projection = nn.Sequential(
            nn.Linear(self.embedding_length * 2, self.embedding_length),
            #nn.BatchNorm1d(self.embedding_length),
            #nn.SELU()
            nn.LeakyReLU(),
            #nn.GELU(),
            #nn.LayerNorm(self.embedding_length),
            #nn.BatchNorm1d(self.embedding_length),
            #nn.Dropout(p=0.4),
            #nn.ReLU(),
        )
    
        self.projection = projection.to(device)
        #classification_dim = min([_.out_features for _ in projection.modules() if isinstance(_, nn.Linear)])
        # assert classification_dim == 512
        classification_dim = self.embedding_length #// 2
        print(f"Classification_dim: {classification_dim}")

        # binary classification head
        head = nn.Sequential(
            nn.Linear(classification_dim, nn_classes),
        )
    

        self.head = head.to(device)

    
    def forward(self, x):

        vh = x['VH']
        vl = x['VL']
        xvh = self.encoder(vh)
        xvl = self.encoder(vl)
        x = torch.fmax(xvh, xvl, out=None)
        x = torch.cat((xvh, xvl), 1)

        return xvh


'''if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaselineOne(1, device, nn_classes=1, freeze_bert=True, model_name='protbert', train_m=True)
    ab = {'VH': 'AABCDTHBFB', 'VL': 'AABCDTHBFB'}
    out = model(ab)
    print(out)
'''