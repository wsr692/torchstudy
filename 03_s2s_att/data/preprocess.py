import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from hparams import hparams

spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')



def tokenize_fr(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_fr.tokenizer(text)][::-1]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def create_dataset(device):
	SRC = Field(tokenize = tokenize_en, 
			init_token = '<sos>', 
			eos_token = '<eos>', 
			include_lengths = True,
			lower = True)

	TRG = Field(tokenize = tokenize_fr, 
			init_token = '<sos>', 
			eos_token = '<eos>', 
			#include_lengths = True,
			lower = True)


	train_data, valid_data, test_data = Multi30k.splits(exts = ('.en', '.fr'), 
													fields = (SRC, TRG))
	SRC.build_vocab(train_data, min_freq = 2)
	TRG.build_vocab(train_data, min_freq = 2)

	#여기에 안 넣는 방법?
	hparams.TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

	hparams.INPUT_DIM = len(SRC.vocab)
	hparams.OUTPUT_DIM = len(TRG.vocab)
	
	train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
	(train_data, valid_data, test_data),
	batch_size = hparams.BATCH_SIZE,
	device = device,
	sort_within_batch = True,
	sort_key = lambda x : len(x.src)
	)

	return train_iterator,valid_iterator,test_iterator,hparams

if __name__=='__main__':
	train_iterator,valid_iterator,test_iterator,hparams,=create_dataset()