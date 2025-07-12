#--------------------------------
# Author : Serge Zaugg
# Description : Testing for CI - basic tests to check if process runs through, nothing more !
# mini-cheat-sheet
# install build:        pip install --upgrade dist\train_saec-0.1.3-py3-none-any.whl
# to work in dev        pip install --upgrade -e .
#                       pip uninstall train_saec
#--------------------------------

import gc
import torch
import plotly
import os
import glob

# check where from pkg was imported
import train_saec.tools
pkg_import_source = train_saec.tools.__file__

from train_saec.tools import MakeColdAutoencoders, AutoencoderTrain, EvaluateReconstruction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dir = "dev/outp/model_dir"
dat_tra_dir = "dev/data/train/images"
dat_tes_dir = "dev/data/test/images" 

#----------------------------------------------
# create objects to be tested

mca = MakeColdAutoencoders(dir_models = model_dir)
mod_arch = mca.make()
# mod_arch.keys()

# (1) try an run each defined model

at01 = AutoencoderTrain(model_dir, dat_tra_dir, dat_tes_dir, False, "conv_tran_L5_TP32", 'baseline', device)
fig01 = at01.make_data_augment_examples(batch_size = 4)
li011, li012, tstmp01 = at01.train_autoencoder(n_epochs = 1, batch_size_tr = 2, batch_size_te = 2, devel = True)
del(at01)
gc.collect()

at02 = AutoencoderTrain(model_dir, dat_tra_dir, dat_tes_dir, False, "conv_tran_L5_TP32_ch512", 'daugm_denoise', device)
fig02 = at02.make_data_augment_examples(batch_size = 5)
li021, li022, _ = at02.train_autoencoder(n_epochs = 2, batch_size_tr = 3, batch_size_te = 1, devel = True)
del(at02)
gc.collect()

at03 = AutoencoderTrain(model_dir, dat_tra_dir, dat_tes_dir, False, "conv_tran_L5_sym", 'daugm_only', device)
li031, li032, _ = at03.train_autoencoder(n_epochs = 1, batch_size_tr = 4, batch_size_te = 2, devel = True)
del(at03)
gc.collect()

at04 = AutoencoderTrain(model_dir, dat_tra_dir, dat_tes_dir, False, "conv_conv_L5_TP32", 'denoise_only', device)
li041, li042, _ = at04.train_autoencoder(n_epochs = 2, batch_size_tr = 2, batch_size_te = 3, devel = True)
del(at04)
gc.collect()

at05 = AutoencoderTrain(model_dir, dat_tra_dir, dat_tes_dir, False, "conv_tran_L4_TP08", 'daugm_denoise', device)
li051, li052, _ = at05.train_autoencoder(n_epochs = 1, batch_size_tr = 2, batch_size_te = 4, devel = True)
del(at05)
gc.collect()

# (2) assess other methods 

# run from pretrained 
at03 = AutoencoderTrain(model_dir, dat_tra_dir, dat_tes_dir, True, tstmp01, 'denoise_only', device)
fig03 = at03.make_data_augment_examples(batch_size = 6)
_, _, tstmp03 = at03.train_autoencoder(n_epochs = 1, batch_size_tr = 2, batch_size_te = 2, devel = True)
del(at03)
gc.collect()

er = EvaluateReconstruction(dir_models = model_dir, device = device)
fig_reconst = er.evaluate_reconstruction_on_examples(dat_tes_dir, tstmp03, n_images = 8, shuffle = False)


#----------------------------------------------
# perform the tests
# mostly to see if process runs through - not fine grained unit tests

def test_import_location():  
    """ not really a test, this will print the pkg_import_source with $ pytest -s """  
    print('-->> pkg_import_source: ' , pkg_import_source)

def test_MakeColdAutoencoders_001():
    assert isinstance(mca, MakeColdAutoencoders)

def test_MakeColdAutoencoders_002():
    assert len(mod_arch) == 5

def test_AutoencoderTrain_01():
    assert len(li011) == 1
    assert len(li012) == 1

def test_AutoencoderTrain_02():
    assert len(li021) == 2
    assert len(li022) == 2

def test_AutoencoderTrain_03():
    assert len(li031) == 1
    assert len(li032) == 1

def test_AutoencoderTrain_03():
    assert len(li041) == 2
    assert len(li042) == 2

def test_AutoencoderTrain_03():
    assert len(li051) == 1
    assert len(li052) == 1

def test_fig_data_augment():
    assert isinstance(fig01, plotly.graph_objs._figure.Figure)
    assert isinstance(fig02, plotly.graph_objs._figure.Figure)
    assert isinstance(fig03, plotly.graph_objs._figure.Figure)

def test_fig_reconstruction():
    assert isinstance(fig_reconst, plotly.graph_objs._figure.Figure)

#----------------------------------------------
# clean-up : Find and Delete all .pth and .pkl files 
npz_files1 = glob.glob(os.path.join(model_dir, '*.pth'))
pkl_files1 = glob.glob(os.path.join(model_dir, '*.pkl'))
files_to_remove = npz_files1 + pkl_files1
len(files_to_remove)
for file_path in files_to_remove:
    os.remove(file_path)
