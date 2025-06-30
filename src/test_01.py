#--------------------------------
# Author : Serge Zaugg
# Description : Code testing for CI
#--------------------------------

import gc
import torch
from train_saec import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction
# from src.train_saec import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction

#----------------------------------------------
# create objects to be tested

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cold_dir = "dev/outp/cold_models"
hot_dir = "dev/outp/hot_models"
dat_tra_dir = "dev/data/train/images"
dat_tes_dir = "dev/data/test/images" 

# Create the cold instances of the models
mca = MakeColdAutoencoders(dir_cold_models = cold_dir)
mod_arch = mca.make()

at01 = AutoencoderTrain(cold_dir, hot_dir, dat_tra_dir, dat_tes_dir, False, "GenBTP32_CH0256", 'baseline', device)
li011, li012 = at01.train_autoencoder(n_epochs = 1, batch_size_tr = 8, batch_size_te = 4, devel = True)
gc.collect()

at01 = AutoencoderTrain(cold_dir, hot_dir, dat_tra_dir, dat_tes_dir, False, "GenC_new_TP32_CH0256", 'daugm_denoise', device)
li021, li022 = at01.train_autoencoder(n_epochs = 2, batch_size_tr = 2, batch_size_te = 3, devel = True)
gc.collect()

#----------------------------------------------
# perform the tests

def test_001():
    assert isinstance(mca, MakeColdAutoencoders)

def test_002():
    assert len(mod_arch) == 6

def test_01():
    assert len(li011) == 1
    assert len(li012) == 1

def test_02():
    assert len(li021) == 2
    assert len(li022) == 2

# # Directly check data augmentation
# at.make_data_augment_examples().show()



# # Or, initialize a AEC-trainer with a pre-trained model
# at = AutoencoderTrain(data_gen = 'daugm_denoise', 
#                       dir_cold_models = cold_dir, 
#                       dir_hot_models = hot_dir,
#                       dir_train_data = dat_tra_dir, 
#                       dir_test_data = dat_tes_dir,
#                       hot_start = True, 
#                       model_tag = "20250630_112654", 
#                       device = device)


# # EvaluateReconstruction
# er = EvaluateReconstruction(dir_hot_models = hot_dir, device = device)
# er.evaluate_reconstruction_on_examples(
#     path_images = dat_tes_dir, 
#     time_stamp_model = "20250630_112752", n_images = 64, shuffle = False).show()

