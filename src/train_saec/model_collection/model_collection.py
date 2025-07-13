#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import torch
import torch.nn as nn

# -------------------------------------------------
# Encoder convolution(freq-pool : 128 - time-pool : 32)
class Encoder_conv_L5_TP32(nn.Module):
    def __init__(self, n_ch_in = 3, n_ch_out = 256, ch = [64, 128, 128, 128]):
        super().__init__()
        po = [(2, 2), (4, 2), (4, 2), (2, 2), (2, 2)]
        self.padding =  "same"
        conv_kernel = (3,3)
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[0],  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0]))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[2], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2]))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[3], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.AvgPool2d(po[3], stride=po[3]))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], n_ch_out, kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(n_ch_out, n_ch_out, kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.AvgPool2d(po[4], stride=po[4]))
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return(x)

# -------------------------------------------------
# Decoder Transpose convolution
class Decoder_tran_L5_TP32(nn.Module):
    def __init__(self, n_ch_in=256, n_ch_out=3, ch = [128, 128, 128, 64]):
        super().__init__()
        po =  [(2, 2), (2, 2), (4, 2), (4, 2), (2, 2)]
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(n_ch_in, ch[0], kernel_size=(5,5), stride=po[0], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(1,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(1,2),  output_padding=(1,1)), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.out_map = nn.Sequential(
            nn.ConvTranspose2d(ch[3], n_ch_out, kernel_size=(5,5), stride=po[4], padding=(2,2),  output_padding=(1,1)), 
            nn.Sigmoid()
            )
   
    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.out_map(x)
        return x

# -------------------------------------------------
# Decoder convolution
class Decoder_conv_L5_TP32(nn.Module):
    def __init__(self, n_ch_in=256, n_ch_out=3, ch = [128, 128, 128, 64]):
        super().__init__()
        self.padding =  "same"
        po =  [(2, 2), (2, 2), (4, 2), (4, 2), (2, 2)]
        self.tconv0 = nn.Sequential(
            nn.Upsample(scale_factor=po[0], mode='bilinear'),
            nn.Conv2d(n_ch_in,  n_ch_in, kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            )
        self.tconv1 = nn.Sequential(
            nn.Upsample(scale_factor=po[1], mode='bilinear'),
            nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            )
        self.tconv2 = nn.Sequential(
            nn.Upsample(scale_factor=po[2], mode='bilinear'),
            nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),    
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            )
        self.tconv3 = nn.Sequential(
            nn.Upsample(scale_factor=po[3], mode='bilinear'),
            nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            )
        self.out_map = nn.Sequential(
            nn.Upsample(scale_factor=po[4], mode='bilinear'),
            nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.Conv2d(ch[3], n_ch_out, kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.out_map(x)
        return x

# -------------------------------------------------
# simpler model (freq-pool : 128 - time-pool : 8)

class Encoder_conv_L4_TP08(nn.Module):
    def __init__(self, n_ch_in = 3, n_ch_out = 256,  ch = [64, 128, 128]):
        super().__init__()
        po = [(2, 2), (2, 2), (2, 2)]
        self.padding =  "same"
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0]))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2]))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], n_ch_out, kernel_size=(16,1), stride=1, padding='valid'),
            )   
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return(x)
    
class Decoder_tran_L4_TP08(nn.Module):
    def __init__(self, n_ch_in = 256, n_ch_out=3, ch = [128, 128, 64]):
        super().__init__()
        po =  [(2, 2), (2, 2), (2, 2)]
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(n_ch_in, ch[0], kernel_size=(16,1), stride=(1,1), padding=(0,0), output_padding=(0,0)), 
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[0], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], n_ch_out, kernel_size=(5,5), stride=po[2], padding=(2,2), output_padding=(1,1)),  
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        return x

# -------------------------------------------------
# Encoder / Decoder  (freq-pool : 32+4 - time-pool : 32)
class Encoder_conv_L5_sym(nn.Module):
    def __init__(self, n_ch_in = 3, n_ch_out = 256, ch = [64, 128, 128, 128]):
        super().__init__()
        po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
        self.padding =  "same"
        conv_kernel = (3,3)
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[0],  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0]))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[2], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2]))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[3], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.AvgPool2d(po[3], stride=po[3]))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], n_ch_out, kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(n_ch_out, n_ch_out, kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(n_ch_out),
            nn.ReLU(),
            nn.AvgPool2d(po[4], stride=po[4]))
        # spec layer to combine the freq to 1 dim
        self.conv_spec = nn.Sequential(
            nn.Conv2d(n_ch_out, n_ch_out, kernel_size=(4,1), stride=1, padding='valid'),
            )
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv_spec(x)
        return(x)

class Decoder_tran_L5_sym(nn.Module):
    def __init__(self, n_ch_in=256, n_ch_out=3, ch = [128, 128, 128, 64]):
        super().__init__()
        po =  [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
        self.tconv_spec = nn.Sequential(
            nn.ConvTranspose2d(n_ch_in, n_ch_in, kernel_size=(4,1), stride=(1,1), padding=(0,0), output_padding=(0,0)), 
            nn.BatchNorm2d(n_ch_in),
            nn.ReLU()
            )
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(n_ch_in, ch[0], kernel_size=(5,5), stride=po[0], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(2,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(2,2),  output_padding=(1,1)), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.out_map = nn.Sequential(
            nn.ConvTranspose2d(ch[3], n_ch_out, kernel_size=(5,5), stride=po[4], padding=(2,2),  output_padding=(1,1)), 
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.tconv_spec(x)
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.out_map(x)
        return x

# # -------------------------------------------------
# # Encoder draft Proposed by ChatGPT + major modifs by Serge, Decoder by Serge:
# # "I need a CNN architecture to extract features for texture classification, what can you recommend ?""
# # "I am using images of size 128 by 256 pixels"

class ReshapeLayer(nn.Module):
    """ array of dim (b, ch, f, t) is reshaped to (b, ch*f, t) , or the reverse"""
    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape  # Shape excluding the batch size
    def forward(self, x):
        return x.view(x.size(0), *self.target_shape, x.size(3))
        # Or use x.reshape if you're worried about non-contiguous tensors:

class Encoder_texture(nn.Module):
    def __init__(self, n_ch_in = 3, n_ch_out = 256):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in, 32, kernel_size=3, padding=1),   
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   )                           
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  )   
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), )
        # older - nice but make huge dim
        # self.spec = ReshapeLayer(128 * 32 , 1)  
        # spec layer to combine the freq to 1 dim
        self.spec = nn.Sequential(
            nn.Conv2d(128, n_ch_out, kernel_size=(32,1), stride=1, padding='valid'),
            )   
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.spec(x)
        return x

class Decoder_texture(nn.Module):
    def __init__(self, n_ch_in = 256, n_ch_out = 3):
        super().__init__()
        # older - nice but make huge dim
        # self.spec = ReshapeLayer(128, 32)
        # spec layer
        self.spec = nn.Sequential(
            nn.ConvTranspose2d(n_ch_in, 128, kernel_size=(32,1), stride=(1,1), padding=(0,0), output_padding=(0,0)), 
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(3,3), stride=1, padding=(1,1), output_padding=(0,0)), 
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=2, padding=(1,1), output_padding=(1,1)), 
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=2, padding=(1,1), output_padding=(1,1)), 
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, n_ch_out, kernel_size=(3,3), stride=1, padding=(1,1), output_padding=(0,0)), 
            nn.Sigmoid())
    def forward(self, x):
        x = self.spec(x)
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        return x



# devel code - supress execution if this is imported as module 
if __name__ == "__main__":

    from torchinfo import summary

    # # conv-tran TP32
    # model_enc = Encoder_conv_L5_TP32(n_ch_in = 3,   n_ch_out = 256, ch = [64, 64, 128, 256])
    # model_dec = Decoder_tran_L5_TP32(n_ch_in = 256, n_ch_out = 3,   ch = [256, 128, 64, 64])
    # summary(model_enc, (1, 3, 128, 1152), depth = 1)
    # summary(model_dec, (1, 256, 1, 36), depth = 1)

    # # conv-conv TP32
    # model_enc = Encoder_conv_L5_TP32(n_ch_in = 3,   n_ch_out = 256, ch = [64, 64, 128, 256])
    # model_dec = Decoder_conv_L5_TP32(n_ch_in = 256, n_ch_out = 3,   ch = [256, 128, 64, 64])
    # summary(model_enc, (1, 3, 128, 1152), depth = 1)
    # summary(model_dec, (1, 256, 1, 36), depth = 1)

    # # conv-tran TP08 
    # model_enc = Encoder_conv_L4_TP08(n_ch_in = 3,   n_ch_out = 256, ch = [64, 64, 128])
    # model_dec = Decoder_tran_L4_TP08(n_ch_in = 256, n_ch_out= 3,    ch = [128, 64, 64])
    # summary(model_enc, (1, 3, 128, 1152), depth = 1)
    # summary(model_dec, (1, 256, 1, 144), depth = 1)

    # conv-tran sym
    model_enc = Encoder_conv_L5_sym(n_ch_in = 3,   n_ch_out = 256, ch = [64, 64, 128, 256])
    model_dec = Decoder_tran_L5_sym(n_ch_in = 256, n_ch_out = 3,   ch = [256, 128, 64, 64])
    summary(model_enc, (1, 3, 128, 1152), depth = 1)
    summary(model_dec, (1, 256, 1, 36), depth = 1)

    model_enc = Encoder_texture(n_ch_in = 3, n_ch_out = 512)
    model_dec = Decoder_texture(n_ch_in = 512, n_ch_out = 3)
    # summary(model_enc, (1, 3, 128, 256), depth = 1)
    # summary(model_dec, (1, 4096, 1, 64), depth = 1)
    summary(model_enc, (1, 3, 128, 1152), depth = 1)
    summary(model_dec, (1, 512, 1,  288), depth = 1)






    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get size of receptive field 
    class Convnet00(nn.Module):
        def __init__(self):
            super(Convnet00, self).__init__()
            ch = 55
            po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
            self.conv0 = nn.Sequential(
                nn.Conv2d(1,  ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[0], stride=po[0]))
            self.conv1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[1], stride=po[1]))
            self.conv2 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[2], stride=po[2]))
            self.conv3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[3], stride=po[3]))
            self.conv4 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[4], stride=po[4]))
            self.conv5 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[5], stride=po[5]))
        def forward(self, x):
            x = self.conv0(x)
            x = self.conv1(x)
            x = self.conv2(x) # till here : pool 8
            # x = self.conv3(x) # till here : pool 16
            # x = self.conv4(x) # till here : pool 32
            # x = self.conv5(x) # till here : pool 64
            return(x)
        
    model_conv = Convnet00() 
    model_conv = model_conv.to(device)
    summary(model_conv, (1, 316, 316)) # pool 64
    summary(model_conv, (1, 156, 156)) # pool 32
    summary(model_conv, (1, 76, 76)) # pool 16
    summary(model_conv, (1, 36, 36)) # pool 8




