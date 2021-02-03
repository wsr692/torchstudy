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

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):    
    epoch_loss = 0
    epoch_acc = 0    
    model.train() 
    for batch in iterator:
        # print(batch.text)
        optimizer.zero_grad()        
        predictions = model(batch.text).squeeze(1)        
        loss = criterion(predictions, batch.label.float())        
        acc = binary_accuracy(predictions, batch.label)        
        loss.backward()
        optimizer.step()        
        epoch_loss += loss.item()
        epoch_acc += acc.item()    

        # #  l2 norm (weight contraints): 3
        # for name, param in model.named_parameters():
        #     if 'fc.weight' in name:
        #         max_val = 3
        #         norm = param.norm( dim=0, keepdim=True)
        #         desired = torch.clamp(norm, 0, max_val)
        #         scale = desired / (1e-7 + norm)
        #         #param *= FloatTensor([scale])

        #         param = param * scale
                
        
        # epoch_loss += loss.item()
        # epoch_acc += acc.item()
        

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):    
    epoch_loss = 0
    epoch_acc = 0    
    model.eval()    
    with torch.no_grad():    
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)            
            loss = criterion(predictions, batch.label.float())            
            acc = binary_accuracy(predictions, batch.label.float())
            epoch_loss += loss.item()
            epoch_acc += acc.item()        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs