class hparams:

	# model
	ENC_EMB_DIM = 256
	DEC_EMB_DIM = 256
	HID_DIM = 512
	N_LAYERS = 2
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5
	
	# training
	BATCH_SIZE=128
	N_EPOCHS=10
	CLIP=1


	# dirs
	save_dir='./ckpt'