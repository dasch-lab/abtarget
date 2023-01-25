import torch
from torch import nn
from transformers import BertModel, BertTokenizer, logging

PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert_bfd"
MAX_LEN = 512
logging.set_verbosity_error()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {0}".format(device))


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

        #x['input_ids']=x['input_ids'].to(device)
        #x['attention_mask']=x['attention_mask'].to(device)

        # Embed sequence
        #output = self.bert(
        #input_ids=x['input_ids'],
        # attention_mask=x['attention_mask']
        #)
        #return output.last_hidden_state

    

class Baseline(nn.Module):
    """
    # Schema
    [BERT-VH]--|
               |--[CONCAT]--[LayerNorm,Linear, Gelu]x2--> Classification (0/1) protein or non-protein
    [BERT-VL]--|
    # Note: criterion = torch.nn.CrossEntropyLoss()
    """

    def __init__(self, batch_size, device, nn_classes=2, freeze_bert=True):
        super().__init__()

        self.dropout = 0.8
        self.embedding_length = 1024
        self.batch_size = batch_size
        self.device = device


        # BertEncoder
        self.bert_encoder = BertEncoder()
        # bert_emb_size = (list(self.bert_encoder.bert.children())[1].layer[-1].output.dense.out_features)

        if freeze_bert:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False

        self.freeze_bert = freeze_bert
        self.softmax = nn.LogSoftmax(dim = 1)

        # Projection for the concatenated embeddings
        
        projection = nn.Sequential(
            nn.Linear(self.embedding_length * 2, self.embedding_length),
            nn.ReLU(),
            #nn.Linear(bert_emb_size, bert_emb_size // 2),
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
            #nn.Sigmoid()
        )
    

        self.head = head.to(device)

    
    def forward(self, x):

        vh = x['VH']
        vl = x['VL']
        xvh = self.bert_encoder(vh)
        xvl = self.bert_encoder(vl)
        x = torch.cat((xvh, xvl), 1)

        x = self.projection(x)
        x = self.head(x)
        #logits = self.label(x)
        #logits = self.softmax(x)

        return x


# Check model parameter freezing
# baseline = Baseline(nn_classes=2, freeze_bert=False)
# list(baseline.children())
# print(sum([param.nelement() for param in baseline.parameters() if param.requires_grad]))
#for name, param in baseline.named_parameters():
#    if param.requires_grad:
#        print('## ', name, param.requires_grad)