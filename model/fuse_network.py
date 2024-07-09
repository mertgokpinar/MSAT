import torch 
import torch.nn as nn



'''
TransformerV1
'''

def normalization(tensor):
    norm = (tensor - torch.min(tensor)/(torch.max(tensor)-torch.min(tensor)))
    return norm

def mean(tensor_1,tensor_2):

    tensor_3 = torch.add(tensor_1,tensor_2)
    tensor_3 = torch.div(tensor_3,2)

    return tensor_3


class Transformer(nn.Module):
    def __init__(self, input_size,dropout):
        super(Transformer,self).__init__()

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_1 = nn.Transformer(d_model=input_size, nhead = 2, num_encoder_layers = 2, num_decoder_layers=2, dropout=dropout) # d_model represents length of features in sequence 
        self.linear = nn.Linear(input_size,input_size)
        self.norm = nn.LayerNorm(32)
        self.relu = nn.ReLU()

    def forward(self,src,tgt):
        '''
        https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        '''
        
        src = normalization(src) 
        tgt = normalization(tgt)
        tgt_1= self.transformer_1 (src,tgt)
        tgt_1= self.linear (tgt_1)     
        return tgt_1

