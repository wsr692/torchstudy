
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets

import random
import numpy as np
import pandas as pd 
import codecs
import pickle,re
import time

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx,pretrained_embedding):
        
        super().__init__()
        
        self.nonstat_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.nonstat_emb.from_pretrained(pretrained_embedding.clone().detach(),max_norm=3.0, freeze=False)
        self.static_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.static_emb=self.static_emb.from_pretrained(pretrained_embedding.clone().detach())


        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim),
                                padding=(filter_sizes[0]-1, 0))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim),
                                padding=(filter_sizes[1]-1, 0))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim),
                                padding=(filter_sizes[2]-1, 0))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        nonstat_emb = self.nonstat_emb(text)
        static_emb= self.static_emb(text)

        #embedded = [batch size, sent len, emb dim]
        nonstat_emb = nonstat_emb.unsqueeze(1)
        static_emb = static_emb.unsqueeze(1)

        conved_0 = F.relu(self.conv_0(nonstat_emb).squeeze(3))+F.relu(self.conv_0(static_emb).squeeze(3))
        conved_1 = F.relu(self.conv_1(nonstat_emb).squeeze(3))+F.relu(self.conv_1(static_emb).squeeze(3))
        conved_2 = F.relu(self.conv_2(nonstat_emb).squeeze(3))+F.relu(self.conv_2(static_emb).squeeze(3))
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):        
        super().__init__()        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)        
        self.dropout = nn.Dropout(dropout)        
    def forward(self, text):        
        #text = [batch size, sent len]        
        embedded = self.embedding(text)                
        #embedded = [batch size, sent len, emb dim]        
        embedded = embedded.permute(0, 2, 1)        
        #embedded = [batch size, emb dim, sent len]        
        conved = [F.relu(conv(embedded)) for conv in self.convs]            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]        
        #pooled_n = [batch size, n_filters]        
        cat = self.dropout(torch.cat(pooled, dim = 1))        
        #cat = [batch size, n_filters * len(filter_sizes)]            
        return self.fc(cat)

