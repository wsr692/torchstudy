
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.model_selection import KFold
import random
import numpy as np
import pandas as pd 
import codecs
import pickle,re
import time

from model.cnn import CNN1d
from data.mrdataset import MrData
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__=='__main__':

    pos_src='data/raw/pos.txt'
    neg_src='data/raw/neg.txt'
    result_path='data/raw/data.txt'
    split_ratio_train_test=0.9
    split_ratio_train_val=7./9

    SEED = 1234
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)

    # get data as iterators
    custom_data=MrData()
    custom_data.prep_tabular(pos_src,neg_src,result_path)
    TEXT,train_iterator,valid_iterator,test_iterator,train_data,test_data,dataset=custom_data.tabular2iter(result_path,split_ratio_train_test,split_ratio_train_val,SEED)
    
    #print("dataset",type(dataset))
    dataset = ConcatDataset([dataset])

    # TEXT 필요
    # model config
    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    pretrained_embeddings = TEXT.vocab.vectors
    N_FILTERS = 100
    FILTER_SIZES = [3,4,5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    EMBEDDING_DIM = 300

    # Initialize a model here...
    model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # train
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 20
    best_valid_loss = float('inf')

    #dataset = ConcatDataset([train_data,test_data])
    k_folds=10
      # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset,batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,batch_size=10, sampler=test_subsampler)
        print(trainloader,type(trainloader))
        #train_iterator=data.Iterator(train_data, batch_size=, train=True, device=device, sort_key=lambda x: len(x.text), sort_within_batch=False)
        #test_iterator = data.Iterator(test_data, batch_size=self.batch_size, train=False, device=device, sort_key=lambda x: len(x.text), sort_within_batch=False)


        for epoch in range(N_EPOCHS):
            print(f'FOLD {fold} / {k_folds} EPOCH {epoch} / {N_EPOCHS}')
            start_time = time.time()

            train_loss, train_acc = train(model, trainloader, optimizer, criterion)
            #train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'model.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        # evaluate
        model.load_state_dict(torch.load('model.pt'))
        test_loss, test_acc = evaluate(model, testloader, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')