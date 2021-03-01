import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import math
import time

class Seq2SeqTrain():
	def __init__(self,model,hparams):
		self.model=model
		self.optimizer = optim.Adam(self.model.parameters())
		self.criterion = nn.CrossEntropyLoss(ignore_index = hparams.TRG_PAD_IDX)
		self.hparams=hparams

	def init_weights(self,m):
		for name, param in m.named_parameters():
			nn.init.uniform_(param.data, -0.08, 0.08)

	def count_parameters(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

	def train(self,iterator):
		
		self.model.train()
		
		epoch_loss = 0
		
		for i, batch in enumerate(iterator):
			
			src = batch.src
			trg = batch.trg
			
			self.optimizer.zero_grad()
			
			output = self.model(src, trg)
			
			#trg = [trg len, batch size]
			#output = [trg len, batch size, output dim]
			
			output_dim = output.shape[-1]
			
			output = output[1:].view(-1, output_dim)
			trg = trg[1:].view(-1)
			
			#trg = [(trg len - 1) * batch size]
			#output = [(trg len - 1) * batch size, output dim]
			
			loss = self.criterion(output, trg)
			
			loss.backward()
			
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.CLIP)
			
			self.optimizer.step()
			
			epoch_loss += loss.item()
			
		return epoch_loss / len(iterator)

	def evaluate(self,model,iterator):
		
		model.eval()
		
		epoch_loss = 0
		
		with torch.no_grad():
		
			for i, batch in enumerate(iterator):

				src = batch.src
				trg = batch.trg

				output = model(src, trg, 0) #turn off teacher forcing

				#trg = [trg len, batch size]
				#output = [trg len, batch size, output dim]

				output_dim = output.shape[-1]
				
				output = output[1:].view(-1, output_dim)
				trg = trg[1:].view(-1)

				#trg = [(trg len - 1) * batch size]
				#output = [(trg len - 1) * batch size, output dim]

				loss = self.criterion(output, trg)
				
				epoch_loss += loss.item()
			
		return epoch_loss / len(iterator)

	def epoch_time(self,start_time, end_time):
		elapsed_time = end_time - start_time
		elapsed_mins = int(elapsed_time / 60)
		elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
		return elapsed_mins, elapsed_secs

	def run(self,train_iterator,valid_iterator):
		print(f'The model has {self.count_parameters():,} trainable parameters')

		self.model.apply(self.init_weights)

		best_valid_loss = float('inf')

		for epoch in range(self.hparams.N_EPOCHS):
			
			start_time = time.time()
			
			train_loss = self.train(train_iterator)
			valid_loss = self.evaluate(valid_iterator)
			
			end_time = time.time()
			
			epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
			
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				torch.save(self.model.state_dict(), f'{self.hparams.save_dir}/fr2en.pt')
			
			print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
			print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
			print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

