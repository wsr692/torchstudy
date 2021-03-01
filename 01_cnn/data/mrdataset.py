import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np
import pandas as pd 
import codecs
import pickle,re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.backends.cudnn.deterministic = True

class MrData():

    def __init__(self,batch_size=64):
        self.batch_size=batch_size

        # load previously saved word vectors
        w2v_saved='data/w2v.pkl'
        with open(w2v_saved,'rb') as f:
            self.w2v=pickle.load(f)
        self.w2idx = {token: token_index for token_index, token in enumerate(self.w2v.index2word)}

    def clean_str(self,string):
        return string.strip()

    def prep_tabular(self,pos_src,neg_src,result_path):
        pos_src='data/raw/pos.txt'
        neg_src='data/raw/neg.txt'
        with open(pos_src, 'r', encoding = "ISO-8859-1") as f:
            pos = [(1,p) for p in f.readlines()]
        with open(neg_src, 'r', encoding = "ISO-8859-1") as f:
            neg = [(0,n) for n in f.readlines()]

        mrdata = pos + neg
        random.shuffle(mrdata)
        result_path='data/raw/data.txt'
        writer = codecs.open(result_path, 'w', encoding='utf-8')
        for line in mrdata:
            cleaned=self.clean_str(line[1])
            writer.write(f"{line[0]}\t{cleaned}\n")
        writer.close()

    def tabular2iter(self,tabular_path,split_ratio_train_test,split_ratio_train_val,SEED):
        # define fields
        TEXT = data.Field(sequential=True, 
                            use_vocab=True,
                            tokenize=lambda x: x.split(' '),
                            lower=False,
                            batch_first=True,
                            fix_length=20)
        LABEL = data.Field(sequential=False, unk_token=None)



        dataset = data.TabularDataset(tabular_path, fields=[('label', LABEL), ('text', TEXT)], format='csv', 
                                    csv_reader_params={'delimiter': '\t'}, skip_header=True)

        train_data, test_data=dataset.split(split_ratio=split_ratio_train_test,random_state=random.seed(SEED))
        train_data, valid_data = train_data.split(split_ratio=split_ratio_train_val, random_state = random.seed(SEED))



        TEXT.build_vocab(train_data,max_size=15000)
        LABEL.build_vocab(train_data)
        TEXT.vocab.set_vectors(self.w2idx, torch.from_numpy(self.w2v.vectors).float().to(device), self.w2v.vector_size)

        
        # create data iterators
        train_iterator = data.Iterator(train_data, batch_size=self.batch_size, train=True, device=device, sort_key=lambda x: len(x.text), sort_within_batch=False)
        valid_iterator = data.Iterator(valid_data, batch_size=self.batch_size, train=False, device=device, sort_key=lambda x: len(x.text), sort_within_batch=False)
        test_iterator = data.Iterator(test_data, batch_size=self.batch_size, train=False, device=device, sort_key=lambda x: len(x.text), sort_within_batch=False)

        print(f"train {len(train_iterator)} valid {len(valid_iterator)} test_iterator {len(test_iterator)}")


        return TEXT,train_iterator,valid_iterator,test_iterator,train_data,test_data,dataset

	# from https://github.com/long8v/torch_study/blob/82d645e30eea414e971a0e986c8ab1172c237d89/paper/01_CNN/source/dataloader.py#L83
    def get_pretrained_embeddings(self):
        pretrained_embedding = []
        for word in self.vocab.stoi_dict:
            if word in self.w2v:
                pretrained_embedding.append(self.w2v[word])
            else: 
                pretrained_embedding.append(np.random.uniform(-0.25, 0.25, 300))
        return torch.from_numpy(np.array(pretrained_embedding))   


# if __name__ == "__main__":
