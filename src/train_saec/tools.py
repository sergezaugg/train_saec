#--------------------------------
# Author : Serge Zaugg
# Description : ML processes are wrapped into functions/classes here
#--------------------------------

import os 
import pickle
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import json
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as transforms
import torch.optim as optim
from torchinfo import summary
from importlib.resources import files
# import own items
import train_saec.model_collection.model_collection as allmodels


print("kangourou-turqoise")


class MakeColdAutoencoders:
    """
    Class for creating and saving multiple 'cold' (untrained) autoencoder architectures.
    This class loads configuration from a YAML file, instantiates several encoder-decoder 
    architecture variants from model_collection, summarizes their structures, and saves 
    them as untrained model files for further use or inspection.
    """

    def __init__(self, dir_models):
        """
        """
        self.dir_models = dir_models

    def make(self):
        """
        This method creates several pairs of encoders and decoders with different 
        architectural variants, summarizes their model structures, and saves the 
        untrained (cold) models to disk as .pth files in the configured directory.

        Returns:
            dict: A dictionary containing model architecture summaries for all 
                generated encoder and decoder pairs. The keys are model names, 
                and the values are dicts with 'enc' and 'dec' summaries.
        """
        arch_di = {}

        #--------------------------------
        # primary models 

        # REFERENCE model standard (256)
        Encoder = allmodels.Encoder_conv_L5_TP32
        Decoder = allmodels.Decoder_tran_L5_TP32
        save_file_name = "conv_tran_L5_TP32_ch256"
        model_enc = Encoder(n_ch_in = 3,   n_ch_out  = 256, ch = [64, 64, 128, 256])
        model_dec = Decoder(n_ch_in = 256, n_ch_out =    3, ch = [256, 128, 64, 64])
        arch_di[save_file_name] = {}
        arch_di[save_file_name]['enc'] = summary(model_enc, (1, 3, 128, 1152), depth = 1)
        arch_di[save_file_name]['dec'] = summary(model_dec, (1, 256, 1, 36), depth = 1)
        torch.save(model_enc, os.path.join(self.dir_models, 'cold_encoder_' + save_file_name + '.pth'))
        torch.save(model_dec, os.path.join(self.dir_models, 'cold_decoder_' + save_file_name + '.pth'))

        # sort-of-SYMMETRIC model 256
        Encoder = allmodels.Encoder_conv_L5_sym
        Decoder = allmodels.Decoder_tran_L5_sym
        save_file_name = "conv_tran_L5_sym"
        model_enc = Encoder(n_ch_in = 3,   n_ch_out  = 256, ch = [64, 64, 128, 256])
        model_dec = Decoder(n_ch_in = 256, n_ch_out =    3, ch = [256, 128, 64, 64])
        arch_di[save_file_name] = {}
        arch_di[save_file_name]['enc'] = summary(model_enc, (1, 3, 128, 1152), depth = 1)
        arch_di[save_file_name]['dec'] = summary(model_dec, (1, 256, 1, 36), depth = 1)
        torch.save(model_enc, os.path.join(self.dir_models, 'cold_encoder_' + save_file_name + '.pth'))
        torch.save(model_dec, os.path.join(self.dir_models, 'cold_decoder_' + save_file_name + '.pth'))

        # REFERENCE model larger (512)
        Encoder = allmodels.Encoder_conv_L5_TP32
        Decoder = allmodels.Decoder_tran_L5_TP32
        save_file_name = "conv_tran_L5_TP32_ch512"
        model_enc = Encoder(n_ch_in = 3,   n_ch_out  = 512, ch = [64, 64, 128, 256])
        model_dec = Decoder(n_ch_in = 512, n_ch_out =    3, ch = [256, 128, 64, 64])
        arch_di[save_file_name] = {}
        arch_di[save_file_name]['enc'] = summary(model_enc, (1, 3, 128, 1152), depth = 1)
        arch_di[save_file_name]['dec'] = summary(model_dec, (1, 512, 1, 36), depth = 1)
        torch.save(model_enc, os.path.join(self.dir_models, 'cold_encoder_' + save_file_name + '.pth'))
        torch.save(model_dec, os.path.join(self.dir_models, 'cold_decoder_' + save_file_name + '.pth'))

        # without transpose conv
        Encoder = allmodels.Encoder_conv_L5_TP32 
        Decoder = allmodels.Decoder_conv_L5_TP32
        save_file_name = "conv_conv_L5_TP32"
        model_enc = Encoder(n_ch_in = 3,   n_ch_out = 256, ch = [64, 64, 128, 256])
        model_dec = Decoder(n_ch_in = 256, n_ch_out =   3, ch = [256, 128, 64, 64])
        arch_di[save_file_name] = {}
        arch_di[save_file_name]['enc'] = summary(model_enc, (1, 3, 128, 1152), depth = 1)
        arch_di[save_file_name]['dec'] = summary(model_dec, (1, 256, 1, 36), depth = 1)
        torch.save(model_enc, os.path.join(self.dir_models, 'cold_encoder_' + save_file_name + '.pth'))
        torch.save(model_dec, os.path.join(self.dir_models, 'cold_decoder_' + save_file_name + '.pth'))

        # simpler model (freq-pool : 128 - time-pool : 8)
        Encoder = allmodels.Encoder_conv_L4_TP08 # Encoder_conv_L4_TP08
        Decoder = allmodels.Decoder_tran_L4_TP08
        save_file_name = "conv_tran_L4_TP08"
        model_enc = Encoder(n_ch_in = 3,   n_ch_out = 256, ch = [64, 64, 128])
        model_dec = Decoder(n_ch_in = 256, n_ch_out =   3, ch = [128, 64, 64])
        arch_di[save_file_name] = {}
        arch_di[save_file_name]['enc'] = summary(model_enc, (1, 3, 128, 1152), depth = 1)
        arch_di[save_file_name]['dec'] = summary(model_dec, (1, 256, 1, 144), depth = 1)
        torch.save(model_enc, os.path.join(self.dir_models, 'cold_encoder_' + save_file_name + '.pth'))
        torch.save(model_dec, os.path.join(self.dir_models, 'cold_decoder_' + save_file_name + '.pth'))

        # shallower model for Textures - hmmm  
        Encoder = allmodels.Encoder_texture 
        Decoder = allmodels.Decoder_texture
        save_file_name = "conv_tran_texture_01"
        model_enc = Encoder()
        model_dec = Decoder()
        arch_di[save_file_name] = {}
        arch_di[save_file_name]['enc'] = summary(model_enc, (1, 3, 128, 1152), depth = 1)
        arch_di[save_file_name]['dec'] = summary(model_dec, (1, 4096, 1, 288), depth = 1)
        torch.save(model_enc, os.path.join(self.dir_models, 'cold_encoder_' + save_file_name + '.pth'))
        torch.save(model_dec, os.path.join(self.dir_models, 'cold_decoder_' + save_file_name + '.pth'))

        return(arch_di)


class SpectroImageDataset(Dataset):
    """
    PyTorch Dataset for spectrogram images with optional denoising and augmentation.
    Loads PNG images from a directory and returns two versions (x_1, x_2) per sample, 
    each optionally denoised and/or augmented, along with the image filename.
    """

    def __init__(self, imgpath, par = None, augment_1=False, augment_2=False, denoise_1=False, denoise_2=False):
        """
        Initialize the SpectroImageDataset.

        Parameters
        ----------
        imgpath : str
            Directory containing PNG images.
        par : dict, optional
            Parameters for augmentation (par['da']) and denoising (par['den']).
        augment_1, augment_2 : bool, optional
            Apply augmentation to x_1/x_2.
        denoise_1, denoise_2 : bool, optional
            Apply denoising to x_1/x_2.

        Notes
        -----
        If any augmentation or denoising is enabled, constructs a torchvision transforms pipeline
        using parameters from `par['da']`. Only PNG files in `imgpath` are considered as dataset samples.
        """
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
        self.par = par
        self.augment_1 = augment_1
        self.augment_2 = augment_2
        self.denoise_1 = denoise_1
        self.denoise_2 = denoise_2

        if self.augment_1 or self.augment_2 or self.denoise_1 or self.denoise_2:
            self.dataaugm = transforms.Compose([
                transforms.RandomAffine(translate=(self.par['da']['trans_prop'], 0.0), degrees=(-self.par['da']['rot_deg'], self.par['da']['rot_deg'])),
                transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = self.par['da']['gnoisesigm'], clip=True),]), p=self.par['da']['gnoiseprob']),
                transforms.ColorJitter(brightness = self.par['da']['brightness'] , contrast = self.par['da']['contrast']),
                ])
 
    def __getitem__(self, index):   
        """
        Returns:
            tuple: (x_1, x_2, y)
                x_1, x_2 (Tensor): Processed images.
                y (str): Filename.
        """  
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))
        # load pimage and set range to [0.0, 1.0]
        x_1 = pil_to_tensor(img).to(torch.float32) / 255.0
        x_2 = pil_to_tensor(img).to(torch.float32) / 255.0
        # simple de-noising with threshold
        # take random thld between 0.0 and self.par['den']['thld']
        if self.denoise_1: 
            denoize_thld = np.random.uniform(low=self.par['den']['thld_lo'], high=self.par['den']['thld_up'], size=1).item()
            x_1[x_1 < denoize_thld ] = 0.0
        if self.denoise_2:
            denoize_thld = np.random.uniform(low=self.par['den']['thld_lo'], high=self.par['den']['thld_up'], size=1).item() 
            x_2[x_2 < denoize_thld ] = 0.0    
        # data augmentation 
        if self.augment_1: 
            x_1 = self.dataaugm(x_1)  
        if self.augment_2:
            x_2 = self.dataaugm(x_2) 
        # prepare meta-data 
        y = self.all_img_files[index]

        return (x_1, x_2, y)
    
    def __len__(self):
        """Number of images in the dataset."""
        return (len(self.all_img_files))


class AutoencoderTrain:
    """
    Handles setup, training, and evaluation of an autoencoder for spectrogram images.

    Attributes
    ----------
    sess_info : dict
        Session configuration parameters.
    train_dataset, test_dataset : SpectroImageDataset
        Datasets for training and testing.
    device : str or torch.device
        Device for computation.
    conf : dict
        Project-wide configuration from YAML.
    model_enc, model_dec : torch.nn.Module
        Encoder and decoder models.
    epoch_restart_value : int
        Epoch to resume from (for hot start).
    """
  
    def __init__(self, dir_models, dir_train_data, dir_test_data, hot_start, model_tag, data_gen, device):
        """
        Initialize session, datasets, models, and config.
        Parameters
        ----------
        device : str or torch.device
            Device for model training ("cpu" or "cuda").
        """
        self.dir_models = dir_models
        self.dir_train_data = dir_train_data
        self.dir_test_data  = dir_test_data
        self.hot_start = hot_start
        self.model_tag = model_tag
        self.device = device

        path_json = "train_saec.data_gen_presets"
       
        # load json 
        with files(path_json).joinpath(data_gen + '.json').open("r") as f:
            sess_info = json.load(f)
        self.sess_info = sess_info    
        self.train_dataset = SpectroImageDataset(self.dir_train_data, par = self.sess_info['data_generator'], augment_1 = True, denoise_1 = False, augment_2 = False, denoise_2 = True)
        self.test_dataset  = SpectroImageDataset(self.dir_test_data,  par = self.sess_info['data_generator'], augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)
        
        if self.hot_start == False:
            tstmp_0 = self.model_tag
            path_enc = [a for a in os.listdir(self.dir_models) if tstmp_0 in a and 'cold_encoder' in a][0]
            path_dec = [a for a in os.listdir(self.dir_models) if tstmp_0 in a and 'cold_decoder' in a][0]
            self.model_enc = torch.load(os.path.join(self.dir_models, path_enc), weights_only = False)
            self.model_dec = torch.load(os.path.join(self.dir_models, path_dec), weights_only = False)
            self.model_enc = self.model_enc.to(self.device)
            self.model_dec = self.model_dec.to(self.device)
            sess_info['model_gen'] = self.model_tag
            self.epoch_restart_value = 0
        elif self.hot_start == True:
            tstmp_1 = self.model_tag
            path_enc = [a for a in os.listdir(self.dir_models) if tstmp_1 in a and 'encoder_model' in a][0]
            path_dec = [a for a in os.listdir(self.dir_models) if tstmp_1 in a and 'decoder_model' in a][0]
            self.model_enc = torch.load(os.path.join(self.dir_models, path_enc), weights_only = False)
            self.model_dec = torch.load(os.path.join(self.dir_models, path_dec), weights_only = False)
            self.model_enc = self.model_enc.to(self.device)
            self.model_dec = self.model_dec.to(self.device) 
            # load info from previous training session 
            path_sess = [a for a in os.listdir(self.dir_models) if tstmp_1 in a and '_session_info' in a][0]
            with open(os.path.join(self.dir_models, path_sess), 'rb') as f:
                self.di_origin_sess = pickle.load(f)
            # load model generation 
            sess_info['model_gen'] = self.di_origin_sess['sess_info']['model_gen']
            self.epoch_restart_value = self.di_origin_sess['epoch'] + 1

        else:
            print("something is wrong with 'hot_start'")
        # return(model_enc, model_dec)


    def make_data_augment_examples(self, batch_size = 12):
        """
        Returns a Plotly figure showing a batch of original and augmented images.

        Parameters
        ----------
        batch_size : int
            Number of image pairs to display.

        Returns
        -------
        fig : plotly Figure
            Visualization of data augmentation.
        """
        pt_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,  shuffle=False, drop_last=True)
        # take only first batch 
        for i, (da_1, da_2, fi) in enumerate(pt_loader, 0):
            if i > 0: break
        fig = make_subplots(rows=batch_size, cols=2)
        for ii in range(batch_size): 
            # print(ii)
            img_1 = da_1[ii].cpu().detach().numpy()
            # img_1 = img_1.squeeze() # 1 ch
            img_1 = np.moveaxis(img_1, 0, 2) # 3 ch
            img_1 = 255*img_1 
            img_2 = da_2[ii].cpu().detach().numpy()
            # img_2 = img_2.squeeze()  # 1 ch
            img_2 = np.moveaxis(img_2, 0, 2) # 3 ch
            img_2 = 255*img_2 
            fig.add_trace(px.imshow(img_1).data[0], row=ii+1, col=1)
            fig.add_trace(px.imshow(img_2).data[0], row=ii+1, col=2)
        fig.update_layout(autosize=True,height=400*batch_size, width = 2000)
        return(fig)
    

    def train_autoencoder(self, n_epochs = 1, batch_size_tr = 8, batch_size_te = 32, devel = False):
        """
        Train autoencoder, evaluate on test data, save models and training metadata.
        Parameters
        ----------
        devel : bool
            If True, runs fewer batches per epoch for debugging.
        Returns
        -------
        None
        """

        # train 
        train_loader  = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size_tr,  shuffle=True, drop_last=True)
        test_loader   = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size_te,  shuffle=True, drop_last=True)

        # instantiate loss and optimizer
        criterion = nn.MSELoss() #nn.BCELoss()
        optimizer = optim.Adam(list(self.model_enc.parameters()) + list(self.model_dec.parameters()), lr=0.001)
        # optimizer = optim.SGD(list(self.model_enc.parameters()) + list(self.model_dec.parameters()), lr=0.01, momentum=0.9)

        n_batches_tr = self.train_dataset.__len__() // batch_size_tr
        n_batches_te = self.test_dataset.__len__() // batch_size_te

        mse_test_li = []
        mse_trai_li = []
        for i, epoch in enumerate(range(self.epoch_restart_value, self.epoch_restart_value + n_epochs)):
            print('Epoch (full trainig history): ', epoch +1)
            print(f"Epoch (current training run): {i + 1}/{n_epochs}")
            #----------------
            # Train the model 
            _ = self.model_enc.train()
            _ = self.model_dec.train()
            trai_perf_li = []
            for batch_tr, (da_tr_1, da_tr_2, fi) in enumerate(train_loader, 0):
                if devel and batch_tr > 1:
                    break
                da_tr_1 = da_tr_1.to(self.device)
                da_tr_2 = da_tr_2.to(self.device)
                # reset the gradients 
                optimizer.zero_grad()
                # forward 
                encoded = self.model_enc(da_tr_1)
                # print('encoded.shape', encoded.shape)
                decoded = self.model_dec(encoded)
                # compute the reconstruction loss 
                loss = criterion(decoded, da_tr_2)
                trai_perf_li.append(loss.cpu().detach().numpy().item())
                # compute the gradients
                loss.backward()
                # update the weights
                optimizer.step()
                # feedback every 10th batch
                if batch_tr % 10 == 0:
                    print('TRAINING loss', np.round(loss.item(),5), " --- "  + str(batch_tr+1) + " out of " + str(n_batches_tr) + " batches")
                    print(decoded.cpu().detach().numpy().min().round(3) , decoded.cpu().detach().numpy().max().round(3) )
                    print("-")
            mse_trai_li.append(np.array(trai_perf_li).mean())        

            #----------------------------------
            # Testing the model at end of epoch 
            _ = self.model_enc.eval()
            _ = self.model_dec.eval()
            with torch.no_grad():
                test_perf_li = []
                for batch_te, (da_te_1, da_te_2, fi) in enumerate(test_loader, 0):
                    # quick-fix
                    if devel and batch_te > 1:
                        break
                    if batch_te > 10: 
                        break 
                    da_te_1 = da_te_1.to(self.device)
                    da_te_2 = da_te_2.to(self.device)
                    # forward 
                    encoded = self.model_enc(da_te_1)
                    # encoded.shape
                    decoded = self.model_dec(encoded)
                    # compute the reconstruction loss 
                    loss_test = criterion(decoded, da_te_2)
                    test_perf_li.append(loss_test.cpu().detach().numpy().item())
                    # feedback every 10th batch
                    if batch_te % 10 == 0:
                        print('TEST loss', np.round(loss_test.item(),5), " --- "  + str(batch_te+1) + " out of " + str(n_batches_te) + " batches")
                mse_test_li.append(np.array(test_perf_li).mean())
            
            # reshape performance metrics to a neat lil df
            mse_test = np.array(mse_test_li)
            mse_trai = np.array(mse_trai_li)
            df_test = pd.DataFrame({"mse" : mse_test})
            df_test['role'] = "test"
            df_trai = pd.DataFrame({"mse" : mse_trai})
            df_trai['role'] = "train"
            df_mse = pd.concat([df_test, df_trai], axis = 0)
            df_mse.shape

            # Save the model and all params 
            epoch_tag = '_epo' + str(epoch +1).zfill(3)
            tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_name = tstmp + "_encoder_model_" + self.sess_info['model_gen'] + epoch_tag + ".pth"
            torch.save(self.model_enc, os.path.join(self.dir_models, model_save_name))
            model_save_name = tstmp + "_decoder_model_" + self.sess_info['model_gen'] + epoch_tag+ ".pth"
            torch.save(self.model_dec, os.path.join(self.dir_models, model_save_name))
            # save metadata 
            di_sess = {'df_mse' : df_mse,'sess_info' : self.sess_info, 'epoch' : epoch}
            sess_save_name = tstmp + "_session_info_" + self.sess_info['model_gen'] + epoch_tag + ".pkl"
            with open(os.path.join(self.dir_models, sess_save_name), 'wb') as f:
                pickle.dump(di_sess, f)
            # save TorchScript model for external projects    
            model_save_name = tstmp + "_encoder_script_" + self.sess_info['model_gen'] + epoch_tag + ".pth"
            model_enc_scripted = torch.jit.script(self.model_enc) # Export to TorchScript
            model_enc_scripted.save(os.path.join(self.dir_models, model_save_name))   
            model_save_name = tstmp + "_decoder_script_" + self.sess_info['model_gen'] + epoch_tag + ".pth"
            model_dec_scripted = torch.jit.script(self.model_dec) # Export to TorchScript
            model_dec_scripted.save(os.path.join(self.dir_models, model_save_name)) 
        # temporary - this a quick fix to implement tests     
        return(mse_test_li, mse_trai_li, tstmp)     


class EvaluateReconstruction:

    def __init__(self, dir_models, device):
        self.device = device
        self.dir_models = dir_models

    def evaluate_reconstruction_on_examples(self, path_images, time_stamp_model, n_images = 16, shuffle = True):
        """
        Evaluate the quality of autoencoder reconstructions on a sample of images.
        Loads a batch of images, reconstructs them using the trained autoencoder,
        and plots side-by-side comparisons of original and reconstructed images.
        Args:
            n_images (int, optional): Number of images to sample and display. Default is 16.
            shuffle (bool, optional): Whether to shuffle the dataset when sampling. Default is True.
        Returns:
            plotly.graph_objs._figure.Figure: A plotly figure showing original and reconstructed images.
        """
        # ---------------------
        # (1) load a few images 
        test_dataset = SpectroImageDataset(path_images, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = n_images, shuffle = shuffle)
        for i_test, (data_1, data_2 , _ ) in enumerate(test_loader, 0):
            if i_test > 0: break
            print(data_1.shape)
            print(data_2.shape)
        # ---------------------
        # (2) load models 
        # NEW with TorchScript models 
        path_enc = [a for a in os.listdir(self.dir_models) if time_stamp_model in a and 'encoder_script' in a][0]
        path_dec = [a for a in os.listdir(self.dir_models) if time_stamp_model in a and 'decoder_script' in a][0]
        model_enc = torch.jit.load(os.path.join(self.dir_models, path_enc))
        model_dec = torch.jit.load(os.path.join(self.dir_models, path_dec))
        model_enc = model_enc.to(self.device)
        model_dec = model_dec.to(self.device)
        _ = model_enc.eval()
        _ = model_dec.eval()
        # ---------------------
        # (3) predict 
        data = data_1.to(self.device)
        encoded = model_enc(data).to(self.device)
        decoded = model_dec(encoded).to(self.device)
        # ---------------------
        # plot 
        fig = make_subplots(rows=n_images, cols=2,)
        for ii in range(n_images) : 
            img_orig = data_2[ii].cpu().numpy()
            # img_orig = img_orig.squeeze() # 1 ch
            img_orig = np.moveaxis(img_orig, 0, 2) # 3 ch
            img_orig = 255.0*img_orig  
            img_reco = decoded[ii].cpu().detach().numpy()
            # img_reco = img_reco.squeeze()  # 1 ch
            img_reco = np.moveaxis(img_reco, 0, 2) # 3 ch
            img_reco = 255.0*img_reco   
            _ = fig.add_trace(px.imshow(img_orig).data[0], row=ii+1, col=1)
            _ = fig.add_trace(px.imshow(img_reco).data[0], row=ii+1, col=2)
        _ = fig.update_layout(autosize=True,height=400*n_images, width = 800)
        _ = fig.update_layout(title="Model ID: " + time_stamp_model)
        return(fig)


# devel 
if __name__ == "__main__":
    print(22)


  




