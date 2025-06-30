#--------------------------------
# Author : Serge Zaugg
# Description : Train spectrogram auto-encoders
#--------------------------------

import torch
from src.train_saec import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cold_dir = "dev/outp/cold_models"
hot_dir = "dev/outp/hot_models"

dat_tra_dir = "dev/data/train/images"
dat_tes_dir = "dev/data/test/images" 

# (run once) Create the cold (=random init) instances of the models
mca = MakeColdAutoencoders(dir_cold_models = cold_dir)
mod_arch = mca.make()

# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(
                      data_gen = 'baseline', 
                    #   data_gen = 'daugm_denoise', 
                    #   data_gen = 'daugm_only', 
                      # data_gen = 'denoise_only', 
                      dir_cold_models = cold_dir, 
                      dir_hot_models = hot_dir,
                      dir_train_data = dat_tra_dir, 
                      dir_test_data = dat_tes_dir,
                      hot_start = False, 
                      model_tag = "GenBTP16_CH0256", 
                      device = device
                      )

# Or, initialize a AEC-trainer with a pre-trained model
at = AutoencoderTrain(data_gen = 'daugm_denoise', 
                      dir_cold_models = cold_dir, 
                      dir_hot_models = hot_dir,
                      dir_train_data = dat_tra_dir, 
                      dir_test_data = dat_tes_dir,
                      hot_start = True, 
                      model_tag = "20250630_111557", 
                      device = device)

# Directly check data augmentation
at.make_data_augment_examples().show()
# Start training (.pth files will be saved to disk)
at.train_autoencoder(n_epochs = 3, batch_size_tr = 8, batch_size_te = 32, devel = True)

# EvaluateReconstruction
er = EvaluateReconstruction(dir_hot_models = hot_dir, device = device)
er.evaluate_reconstruction_on_examples(
    path_images = dat_tes_dir, 
    time_stamp_model = "20250630_111557", n_images = 64, shuffle = False).show()

