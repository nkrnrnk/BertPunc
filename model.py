
from torch import nn
from pytorch_pretrained_bert import BertForMaskedLM

class BertPunc(nn.Module):  
    
    def __init__(self, segment_size, output_size, dropout):
        super(BertPunc, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_vocab_size = 30522
        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bert(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(self.bn(x)))
        return x