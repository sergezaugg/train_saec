#--------------------------------
# Author : Serge Zaugg
# Description : Testing for CI - basic tests to check if process runs through, nothing more !
#--------------------------------

import gc
import torch
import plotly
import os
import glob

from train_saec.tools import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cold_dir = "dev/outp/cold_models"
hot_dir = "dev/outp/hot_models"
dat_tra_dir = "dev/data/train/images"
dat_tes_dir = "dev/data/test/images" 

#----------------------------------------------
# create objects to be tested

mca = MakeColdAutoencoders(dir_models = cold_dir)
mod_arch = mca.make()

at01 = AutoencoderTrain(cold_dir, hot_dir, dat_tra_dir, dat_tes_dir, False, "GenBTP32_CH0256", 'baseline', device)
fig01 = at01.make_data_augment_examples(batch_size = 4)
li011, li012, tstmp01 = at01.train_autoencoder(n_epochs = 1, batch_size_tr = 4, batch_size_te = 4, devel = True)
del(at01)
gc.collect()

at02 = AutoencoderTrain(cold_dir, hot_dir, dat_tra_dir, dat_tes_dir, False, "GenC_new_TP32_CH0256", 'daugm_denoise', device)
fig02 = at02.make_data_augment_examples(batch_size = 5)
li021, li022, tstmp02 = at02.train_autoencoder(n_epochs = 2, batch_size_tr = 2, batch_size_te = 3, devel = True)
del(at02)
gc.collect()

at03 = AutoencoderTrain(cold_dir, hot_dir, dat_tra_dir, dat_tes_dir, True, tstmp02, 'denoise_only',device)
fig03 = at03.make_data_augment_examples(batch_size = 6)
li031, li032, tstmp03 = at03.train_autoencoder(n_epochs = 1, batch_size_tr = 3, batch_size_te = 2, devel = True)
del(at03)
gc.collect()

er = EvaluateReconstruction(dir_models = hot_dir, device = device)
fig_reconst = er.evaluate_reconstruction_on_examples(dat_tes_dir, tstmp03, n_images = 8, shuffle = False)

#----------------------------------------------
# perform the tests

def test_MakeColdAutoencoders_001():
    assert isinstance(mca, MakeColdAutoencoders)

def test_MakeColdAutoencoders_002():
    assert len(mod_arch) == 6

def test_AutoencoderTrain_01():
    assert len(li011) == 1
    assert len(li012) == 1

def test_AutoencoderTrain_02():
    assert len(li021) == 2
    assert len(li022) == 2

def test_AutoencoderTrain_03():
    assert len(li031) == 1
    assert len(li032) == 1

def test_fig_data_augment():
    assert isinstance(fig01, plotly.graph_objs._figure.Figure)
    assert isinstance(fig02, plotly.graph_objs._figure.Figure)
    assert isinstance(fig03, plotly.graph_objs._figure.Figure)

def test_fig_reconstruction():
    assert isinstance(fig_reconst, plotly.graph_objs._figure.Figure)

#----------------------------------------------
# clean-up : Find and Delete all .pth and .pkl files 
npz_files1 = glob.glob(os.path.join(cold_dir, '*.pth'))
npz_files2 = glob.glob(os.path.join(hot_dir, '*.pth'))
pkl_files1 = glob.glob(os.path.join(hot_dir, '*.pkl'))
files_to_remove = npz_files1 + npz_files2 + pkl_files1
len(files_to_remove)
for file_path in files_to_remove:
    os.remove(file_path)
