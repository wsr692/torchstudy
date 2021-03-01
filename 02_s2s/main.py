import os
import torch
import torch.nn as nn
import torch.optim as optim


import spacy
import numpy as np

import random
import math
import time

from model import Seq2Seq,Encoder,Decoder
from trainer import Seq2SeqTrain
from data.preprocess import create_dataset



if __name__=='__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	SEED = 1234

	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True


	# prepare data as iterator
	train_iterator,valid_iterator,test_iterator,hparams=create_dataset(device)

	# define model
	enc = Encoder(hparams.INPUT_DIM, hparams.ENC_EMB_DIM, hparams.HID_DIM, hparams.N_LAYERS, hparams.ENC_DROPOUT)
	dec = Decoder(hparams.OUTPUT_DIM, hparams.DEC_EMB_DIM, hparams.HID_DIM, hparams.N_LAYERS, hparams.DEC_DROPOUT)
	model = Seq2Seq(enc, dec, device).to(device)

	# train
	trainer=Seq2SeqTrain(model,hparams)
	# trainer.run(train_iterator,valid_iterator)

	# test
	model.load_state_dict(torch.load(os.path.join(hparams.save_dir,'fr2en.pt')))
	test_loss = trainer.evaluate(model, test_iterator)
	print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

