#--------------------------------
# Author : Serge Zaugg
# Description : Train spectrogram auto-encoders
# mini-cheat-sheet
# install build:            pip install --upgrade dist\train_saec-0.1.4-py3-none-any.whl
# to work in dev:           pip install --upgrade -e .
# conf no-direct-imports:   pip uninstall train_saec
#--------------------------------

import torch
from train_saec.tools import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir   = "dev/outp/model_dir"
dat_tra_dir = "dev/data/train/images"
dat_tes_dir = "dev/data/test/images" 

# (run once) Create the cold (=random init) instances of the models
mca = MakeColdAutoencoders(dir_models = model_dir)
mod_arch = mca.make()
mod_arch.keys()
mod_arch['conv_tran_texture_01']

#----------------------------------------------------
# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(
    data_gen = 'daugm_denoise', 
    dir_models = model_dir, 
	dir_train_data = dat_tra_dir, 
    dir_test_data = dat_tes_dir,
	hot_start = False, 
    model_tag = "conv_tran_texture_01", 
    device = device
	)

# Directly check data augmentation
at.make_data_augment_examples().show()

# Start training (.pth files will be saved to disk)
_, _, tstmp = at.train_autoencoder(n_epochs = 8, batch_size_tr = 8, batch_size_te = 32, devel = False)


#----------------------------------------------------
# Or, initialize a AEC-trainer with a pre-trained model
at = AutoencoderTrain(
    data_gen = 'daugm_denoise',                      
	dir_models = model_dir,           
	dir_train_data = dat_tra_dir, 
    dir_test_data = dat_tes_dir,
	hot_start = True, 
    model_tag = tstmp, 
    device = device
	)

# Resume training 
_, _, tstmp = at.train_autoencoder(n_epochs = 5, batch_size_tr = 8, batch_size_te = 32, devel = False)


#----------------------------------------------------
# EvaluateReconstruction
er = EvaluateReconstruction(dir_models = model_dir, device = device)
er.evaluate_reconstruction_on_examples(path_images = dat_tes_dir, time_stamp_model = tstmp, n_images = 32, shuffle = False).show()

