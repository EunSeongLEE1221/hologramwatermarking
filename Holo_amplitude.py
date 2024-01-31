import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import numpy as np
import tensorflow as tf
from functions import *
from attack_v9_h1111 import *
#from utils import *
import os
import scipy
import time
import random
import h5py
import json
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import matplotlib.pylab as plt

class WM_NN(object):
    def __init__(self, width=128, height=128, kernel1=3, kernel2=7, batch_size=32, d1_lr=0.0001, g1_lr=0.000025, gan2_lr=0.0001, beta1=0.5, beta2=0.9, lambda1=0.1, lambda2=0.1, lambda3=1.0):
        self.width = width
        self.height = height
        self.x_dim = self.width * self.height
        self.batch_size = batch_size
        self.d1_lr = d1_lr
        self.g1_lr = g1_lr
        self.gan2_lr = gan2_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.f = 0.0
        self.r = 1.0
        
    def generator_1_up(self, z1, is_train=True, reuse=False):
        if not is_train :
         reuse = True
        with tf.variable_scope("gen1_up", reuse=reuse):
            z = z1
            # generator network(using InfoGAN)
            net = {}
            # 1st de-conv. layer inp : (batch_size, width/16, height/16, 1)
            net[1] = deconv2D(z, [self.batch_size, self.height//8, self.width//8, 512], self.kernel1, self.kernel1, 2, 2, name='g1_up_deconv1')
            net[1] = batch_norm(net[1], is_train=is_train, name='g1_up_bnorm1')
            net[1] = tf.nn.relu(net[1])
            net[1] = tf.nn.avg_pool(net[1], ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='g1_up_avg1')
            # 2th de-conv. layer inp : (batch_size, width/8, height/8, 128)
            net[2] = deconv2D(net[1], [self.batch_size, self.height//4, self.width//4, 256], self.kernel1, self.kernel1, 2, 2, name='g1_up_deconv2')
            net[2] = batch_norm(net[2], is_train=is_train, name='g1_up_bnorm2')
            net[2] = tf.nn.relu(net[2])
            net[2] = tf.nn.avg_pool(net[2], ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='g1_up_avg2')
            # 3th de-conv. layer inp : (batch_size, width/4, height/4, 64)
            net[3] = deconv2D(net[2], [self.batch_size, self.height//2, self.width//2, 128], self.kernel1, self.kernel1, 2, 2, name='g1_up_deconv3')
            net[3] = batch_norm(net[3], is_train=is_train, name='g1_up_bnorm3')
            net[3] = tf.nn.relu(net[3])
            net[3] = tf.nn.avg_pool(net[3], ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='g1_up_avg3')
            # 4th de-conv. layer inp : (batch_size, width/2, height/2, 32)
            net[4] = deconv2D(net[3], [self.batch_size, self.height, self.width, 1], self.kernel1, self.kernel1, 2, 2, name='g1_up_deconv4')
            net[4] = tf.nn.avg_pool(net[4], ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='g1_up_avg3')
            
            logit = net[4]
            out = net[4]
            return out, logit, net
    
    def generator_1_Emb(self, x1, z1, is_train=True, reuse=False):
        if not is_train :
         reuse = True
        with tf.variable_scope("gen1_Emb", reuse=reuse):
            # generator network(using InfoGAN)
            net_skip1 = {}
            net_skip1[0] = conv2D(x1, 64, self.kernel1, self.kernel1, 1, 1, name='g1_Emb_deconv_x1')
            net_skip1[1] = tf.concat([net_skip1[0], z1], axis=-1)
            # 1st de-conv. layer inp : (self.batch_size, self.height, self.width, 16+1)
            net_skip1[1] = conv2D(net_skip1[1], 64, self.kernel1, self.kernel1, 1, 1, name='g1_Emb_deconv_')
            net_skip1[1] = batch_norm(net_skip1[1], is_train=is_train, name='g1_Emb_bnorm_')
            net_skip1[1] = tf.nn.relu(net_skip1[1])

            net_skip1[2] = conv2D(net_skip1[1], 64, self.kernel1, self.kernel1, 1, 1, name='g1_Emb_deconv1')
            net_skip1[2] = batch_norm(net_skip1[2], is_train=is_train, name='g1_Emb_bnorm1')
            net_skip1[2] = tf.nn.relu(net_skip1[2])
            
            net_skip1[3] = conv2D(net_skip1[2], 64, self.kernel1, self.kernel1, 1, 1, name='g1_Emb_deconv2')
            net_skip1[3] = batch_norm(net_skip1[3], is_train=is_train, name='g1_Emb_bnorm2')
            net_skip1[3] = tf.nn.relu(net_skip1[3])
            
            net_skip1[4] = conv2D(net_skip1[3], 64, self.kernel1, self.kernel1, 1, 1, name='g1_Emb_deconv3')
            net_skip1[4] = batch_norm(net_skip1[4], is_train=is_train, name='g1_Emb_bnorm3')
            net_skip1[4] = tf.nn.relu(net_skip1[4])

            # 2nd de-conv. layer inp : (self.batch_size, self.height, self.width, 4)
            net_skip1[5] = conv2D(net_skip1[4], 1, self.kernel1, self.kernel1, 1, 1, name='g1_Emb_deconv4')
            #net[2] = batch_norm(net[2], is_train=is_train, name='g1_Emb_bnorm4')
            logit = net_skip1[5]
            net_skip1[5] = tf.nn.tanh(logit)
            out = net_skip1[5]
            return out, logit, net_skip1
    
    def generator_2(self, z2, is_train=True, reuse=False):
        if not is_train :
         reuse = True
        with tf.variable_scope("gen2", reuse=reuse):
            z = z2
            # generator network(using InfoGAN)
            net = {}
            # 1st layer inp : (batch_size, height, width, 1)
            net[1] = conv2D(z, 128, self.kernel2, self.kernel2, 2, 2, name='g2_conv1')
            net[1] = batch_norm(net[1], is_train=is_train, name='g2_bnorm1')
            net[1] = tf.nn.relu(net[1])
            # 2th layer inp : (batch_size, height//2, width//2, 4)
            net[2] = conv2D(net[1], 256, self.kernel2, self.kernel2, 2, 2, name='g2_conv2')
            net[2] = batch_norm(net[2], is_train=is_train, name='g_bnorm2')
            net[2] = tf.nn.relu(net[2])
            # 3th layer inp : (batch_size, height//4, width//4, 16)
            net[3] = conv2D(net[2], 512, self.kernel2, self.kernel2, 2, 2, name='g2_conv3')
            net[3] = batch_norm(net[3], is_train=is_train, name='g2_bnorm3')
            net[3] = tf.nn.relu(net[3])
            # 4th layer inp : (batch_size, height//8, width//8, 64) 
            net[4] = conv2D(net[3], 1, self.kernel2, self.kernel2, 2, 2, name='g2_conv4')
            #net[4] = batch_norm(net[4], is_train=is_train, name='g2_bnorm4')
            #net[4] = tf.nn.relu(net[4])
            # last layer inp : (batch_size, height//16, width//16)
            #net[5] = conv2D(net[4], 1, self.kernel2, self.kernel2, 2, 2, name='g2_conv5')
            logit = net[4]
            net[4] = tf.nn.tanh(logit)
            out = net[4]
            return out, logit, net
    
    def Attack(self, x1):

        data_num = self.batch_size
        mode_num = 12
        self.data_per_attack = data_num//mode_num
        y0 = self.Attack_Identify(x1[:self.data_per_attack])

        y1_1 = self.Attack_Gaussian_Filtering(x1[self.data_per_attack:int(self.data_per_attack*1.25)], kernel=3, sigma=0.5)
        y1_2 = self.Attack_Gaussian_Filtering(x1[int(1.25*self.data_per_attack):int(1.5*self.data_per_attack)], kernel=5, sigma=1)
        y1_3 = self.Attack_Gaussian_Filtering(x1[int(1.5*self.data_per_attack):int(1.75*self.data_per_attack)], kernel=7, sigma=1.5)
        y1_4 = self.Attack_Gaussian_Filtering(x1[int(1.75*self.data_per_attack):int(2*self.data_per_attack)], kernel=9, sigma=2)
        y2_1 = self.Attack_Average_Filtering(x1[2*self.data_per_attack:int(2.5*self.data_per_attack)], kernel=5)
        y2_2 = self.Attack_Average_Filtering(x1[int(2.5*self.data_per_attack):3*self.data_per_attack], kernel=3)
        y2_1_ = self.Attack_Median_Filtering(x1[int(3*self.data_per_attack):int(3.5*self.data_per_attack)], kernel=3)
        y2_2_ = self.Attack_Median_Filtering(x1[int(3.5*self.data_per_attack):int(4*self.data_per_attack)], kernel=5)
        
        y3 = self.Attack_Salt_and_Pepper(x1[4*self.data_per_attack:int(6.5*self.data_per_attack)], p=0.1)
        y4 = self.Attack_Gaussian_Noise(x1[int(6.5*self.data_per_attack):7*self.data_per_attack], sigma=0.1)
        y5_1 = self.Attack_Sharpening(x1[7*self.data_per_attack:int(7.5*self.data_per_attack)], 9)
        y5_2 = self.Attack_Sharpening(x1[int(7.5*self.data_per_attack):8*self.data_per_attack], 5)
        #y6 = self.Attack_Rescaling(x1[7*self.data_per_attack:8*self.data_per_attack])
        #y7 = self.Attack_Cropping(x1[8*self.data_per_attack:9*self.data_per_attack])
        y8 = self.Attack_Rotation(x1[8*self.data_per_attack:9*self.data_per_attack])
        #y9 = self.Attack_Row_and_Column_Removal(x1[9*self.data_per_attack:10*self.data_per_attack])
        #y11 = self.Attack_Contrast(x1[10*self.data_per_attack:11*self.data_per_attack])
        #y12 = self.Attack_Gamma_Correction(x1[10*self.data_per_attack:11*self.data_per_attack])
        y13 = self.Attack_JPEG(x1[9*self.data_per_attack:10*self.data_per_attack])
        y14 = self.Attack_Cropping(x1[10*self.data_per_attack:11*self.data_per_attack])
        y15 = self.Attack_Dropout(x1[11*self.data_per_attack:], self.x_1[11*self.data_per_attack:])

        y = tf.concat([y0, y1_1, y1_2, y1_3, y1_4, y2_1, y2_2, y2_1_, y2_2_, y3, y4, y5_1, y5_2, y8, y13, y14, y15], axis=0)
        
        return y

    def FDM_reconstruction(self, hologram, ref_wavelength, reconstruction_distance, slm_pitch):

        Nr,Nc = np.shape(hologram)
        Nr = np.linspace(0, Nr-1, Nr)-Nr/2
        Nc = np.linspace(0, Nc-1, Nc)-Nc/2 
        k, l = np.meshgrid(Nc,Nr)
    
        lz=ref_wavelength*reconstruction_distance
    
        h=(-1j/lz)*np.exp(1j*np.pi/lz*(np.multiply(k, k)*slm_pitch**2 + np.multiply(l, l)*slm_pitch**2))
        factor = np.multiply(h,hologram)
        reconstructed = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(factor))) # Take inverse Fourier transform of the factor

        reconstructed_field = np.abs(reconstructed)

        return reconstructed_field
        
    def ASM_reconstruction(self, hologram, ref_wavelength, reconstruction_distance, slm_pitch):


        M, N = np.shape(hologram)
        Lx = N*slm_pitch

        Ly = M*slm_pitch

        fx = (np.linspace(0, N-1, N)-N/2)*(1/(slm_pitch*N))
        fy = (np.linspace(0, M-1, M)-M/2)*(1/(slm_pitch*M))    
        FX, FY = np.meshgrid(fx, fy)
        # print("fx",fx)
        # print("fy",fy)
    
        a=np.sqrt((1/(ref_wavelength))**2-np.multiply(FX, FX)-np.multiply(FY, FY))   
        H= np.exp(2j*np.pi*reconstruction_distance*a)

        U1 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(hologram)))
        U2 = np.multiply(U1,np.conjugate(H))
        reconstructed = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(U2)))
        reconstructed_field = np.abs(np.asarray(reconstructed))

        return reconstructed_field
    
    def Attack_test(self, x1):

        data_num = x1.get_shape()[0]
        mode_num = 12
        self.data_per_attack = data_num//mode_num
        #y1_1 = self.Attack_Gaussian_Filtering_test(x1[self.data_per_attack:int(self.data_per_attack*1.5)], kernel=3, sigma=0.5)
        #y1_2 = self.Attack_Gaussian_Filtering_test(x1[int(1.5*self.data_per_attack):int(2*self.data_per_attack)], kernel=5, sigma=1)
        #y1_3 = self.Attack_Gaussian_Filtering_test(x1[int(2*self.data_per_attack):int(2.5*self.data_per_attack)], kernel=7, sigma=1.5)
        #y1_4 = self.Attack_Gaussian_Filtering_test(x1[int(2.5*self.data_per_attack):int(3*self.data_per_attack)], kernel=9, sigma=2)
        #y2_1 = self.Attack_Average_Filtering_test(x1[3*self.data_per_attack:int(4*self.data_per_attack)], kernel=5)
        #y2_2 = self.Attack_Average_Filtering_test(x1[int(4*self.data_per_attack):5*self.data_per_attack], kernel=3)
        
        #y7 = self.Attack_Cropping(x1[8*self.data_per_attack:9*self.data_per_attack])

        
        return y7

    def Attack_Identify(self, x1):

        y = x1
        return y

    def Attack_Gaussian_Filtering(self, x1, kernel=5, sigma=1):

        y = Gaussian_filtering(x1, kernel, sigma, name='Attack_01_k%01d_s%01d'%(kernel, sigma)) 
        return y

    def Attack_Average_Filtering(self, x1, kernel=5):

        y = Average_filtering(x1, kernel, name='Attack_02_k%01d'%(kernel))
        return y

    def Attack_Salt_and_Pepper(self, x1, p=0.1):

        y = Salt_and_Pepper(x1, p, name='Attack_03_p%01d'%(0.1))
        return y
            
    def Attack_Gaussian_Noise(self, x1, sigma=0.1):

        y = Gaussian_Noise(x1, sigma, name='Attack_04_s%01d'%(0.1))
        return y 

    def Attack_Sharpening(self, x1, center=9):

        y = Sharpening(x1, center, name='Attack_05_s%01d'%(center))
        return y
    
    def Attack_Sharpening(self, x1, center=9):

        y = Sharpening(x1, center, name='Attack_05_s%01d'%(center))
        return y

    def Attack_Rescaling(self, x1, ratio=0.5):

        y = Rescaling_test(x1, ratio, name='Attack_06_s%01d'%(0.5))
        return y
    
    '''def Attack_Cropping(self, x1, ratio=0.25):

        y = Cropping(x1, ratio, name='Attack_07_s%01d'%(0.25))
        return y '''
    
    def Attack_Rotation(self, x1, angle=30):

        y = Rotation(x1, name='Attack_08')
        return y
    
    def Attack_Row_and_Column_Removal(self, x1):

        y = Row_and_Column_Removal(x1,  name='Attack_09')
        return y
    
    #def Attack_General_affine_tr(self, x1):
        
        #x1 = tf.multiply(tf.divide(tf.add(x1, 1), 2), 255)
        #x1 = tf.clip_by_value(x1, 0, 255)
        #y = General_affine_tr(x1, sigma, name='Attack_10')
        #y = tf.clip_by_value(tf.subtract(tf.divide(y, 127.5), 1), -1, 1) 
        #return y
    
    def Attack_Contrast(self, x1, factor=5):

        y = Contrast(x1, factor, name='Attack_11_s%01d'%(factor))
        return y
    
    def Attack_Gamma_Correction(self, x1, gamma=0.3):

        y = Gamma_Correction(x1, gamma, name='Attack_12_s%01d'%(gamma))
        return y
    
    def Attack_JPEG(self, x1, quality=50):

        y = JPEG(x1, quality, name='Attack_13_s%01d'%(50))
        return y
    
    def Attack_Cropping(self, x1, ratio=0.5):

        y = Cropping(x1, name='Attack_14_s%01d'%(ratio))
        return y

    def Attack_Dropout(self, x1, x2, p=0.3):

        y = Dropout(x1, x2, name='Attack_15')
        return y
    
    def Attack_Median_Filtering(self, x1, kernel):

        y = Median_filter(x1, kernel, name='Attack_16_k%01d'%(kernel))
        return y


    def BER(self, z, x):
        # BER
        z_1_ = tf.add(z, 1)
        z_1_ = tf.divide(z_1_, 2)
        z_1_ = tf.clip_by_value(z_1_, 0, 1)
        z_1_ = tf.round(z_1_)

        G_test_2_ = tf.add(x, 1)
        G_test_2_ = tf.divide(G_test_2_, 2)
        G_test_2_ = tf.clip_by_value(G_test_2_, 0, 1)
        G_test_2_ = tf.round(G_test_2_)

        BER = tf.equal(z_1_, G_test_2_)
        BER = tf.cast(BER, tf.float32)

        NCC = 1-1*BER

        NCC_A0 = NCC[:self.data_per_attack]
        NCC_A1_1 = NCC[1*self.data_per_attack:int(1.5*self.data_per_attack)]
        NCC_A1_2 = NCC[int(1.5*self.data_per_attack):2*self.data_per_attack]
        NCC_A1_3 = NCC[2*self.data_per_attack:int(2.5*self.data_per_attack)]
        NCC_A1_4 = NCC[int(2.5*self.data_per_attack):3*self.data_per_attack]
        NCC_A2_1 = NCC[3*self.data_per_attack:int(4*self.data_per_attack)]
        NCC_A2_2 = NCC[int(4*self.data_per_attack):5*self.data_per_attack]

        NCC_A2_1_ = NCC[5*self.data_per_attack:int(5.5*self.data_per_attack)]
        NCC_A2_2_ = NCC[int(5.5*self.data_per_attack):6*self.data_per_attack]

        NCC_A3 = NCC[6*self.data_per_attack:int(6.5*self.data_per_attack)]
        NCC_A4 = NCC[int(6.5*self.data_per_attack):7*self.data_per_attack]
        NCC_A5_1 = NCC[7*self.data_per_attack:int(7.5*self.data_per_attack)]
        NCC_A5_2 = NCC[int(7.5*self.data_per_attack):8*self.data_per_attack]
        #NCC_A6 = NCC[7*self.data_per_attack:8*self.data_per_attack]
        #NCC_A7 = NCC[8*self.data_per_attack:9*self.data_per_attack]
        NCC_A8 = NCC[8*self.data_per_attack:9*self.data_per_attack]
        NCC_A13 = NCC[9*self.data_per_attack:10*self.data_per_attack]
        #NCC_A12 = NCC[10*self.data_per_attack:11*self.data_per_attack]
        NCC_A14 = NCC[10*self.data_per_attack:11*self.data_per_attack]
        NCC_A15 = NCC[11*self.data_per_attack:]

        return NCC_A0, tf.reduce_mean([NCC_A1_1, NCC_A1_2, NCC_A1_3, NCC_A1_4]), tf.reduce_mean([NCC_A2_1, NCC_A2_2]), tf.reduce_mean([NCC_A2_1_, NCC_A2_2_]), NCC_A3, NCC_A4, tf.reduce_mean([NCC_A5_1, NCC_A5_2]), NCC_A8, NCC_A13, NCC_A14, NCC_A15


    def build_model(self):
        self.x_1 = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, 1], name='Host_ori')
        self.z_1 = tf.placeholder(tf.float32, [self.batch_size, self.height//16, self.width//16, 1], name='WM')
        self.recon_loss = tf.placeholder(tf.float32, name='recon_loss')
        
        def sigmoid_cross_entropy_with_logits(x, y):
              try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
              except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        # Training graph ################################################################################
        # output of D for fake data
        G_up_fake_1, _, net_up_1 =self.generator_1_up(self.z_1, is_train=True, reuse=False)
        self.G_fake_1, _, net_1 =self.generator_1_Emb(self.x_1, G_up_fake_1, is_train=True, reuse=False)

        # Attack 
        self.G_fake_1_ = self.Attack(self.G_fake_1)
        #self.G_fake_1_test = self.Attack_test(self.G_fake_1)

        # output of D for fake data
        self.G_fake_2_, _, _ = self.generator_2(self.G_fake_1_, is_train=True, reuse=False)
        #self.G_fake_2_test, _, _ = self.generator_2(self.G_fake_1_test, is_train=False, reuse=True)

        # loss for G
        self.invisibility = tf.reduce_mean(tf.multiply(tf.subtract(self.G_fake_1, self.x_1), tf.subtract(self.G_fake_1, self.x_1)))
        self.robustness_ = tf.reduce_mean(tf.abs(tf.subtract(self.G_fake_2_, self.z_1)))

        # loss for Embedder
        self.G1_loss_total_ = (self.lambda1*self.invisibility) + (self.lambda2*self.robustness_) + (self.lambda3 *self.recon_loss)

        # loss for Extractor
        self.G2_loss_total_ = self.lambda1*2*self.robustness_

        # optimizers
        t_vars = tf.trainable_variables()
        self.G_vars_1 = [var for var in t_vars if 'gen1' in var.name]
        self.G_vars_2 = [var for var in t_vars if 'gen2' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.G_optim_1_ = tf.train.AdamOptimizer(self.g1_lr, beta1=self.beta1, beta2=self.beta2) \
                            .minimize(self.G1_loss_total_, var_list=self.G_vars_1)
            self.G_optim_2_ = tf.train.AdamOptimizer(self.gan2_lr, beta1=self.beta1, beta2=self.beta2) \
                            .minimize(self.G2_loss_total_, var_list=self.G_vars_2)

        # Test graph ####################################################################################
        G_up_test_1, _, self.G_up_features_1 = self.generator_1_up(self.z_1, is_train=False, reuse=True)
        self.G_test_1, _, self.G_features_1 = self.generator_1_Emb(self.x_1, G_up_test_1, is_train=False, reuse=True)

        # PSNR
        self.psnr_test = tf.reduce_mean(tf.image.psnr(self.G_test_1, self.x_1, max_val=2))

        self.G_test_1_ = self.Attack(self.G_test_1)

        # output of D for fake data
        self.G_test_2_, _, _ = self.generator_2(self.G_test_1_, is_train=False, reuse=True)

        self.NC_WM_t_ , self.NC_WM_t_A1, self.NC_WM_t_A2, self.NC_WM_t_A22,\
        self.NC_WM_t_A3, self.NC_WM_t_A4,  self.NC_WM_t_A5,\
        self.NC_WM_t_A8, self.NC_WM_t_A13, self.NC_WM_t_A14, self.NC_WM_t_A15 \
         = self.BER(self.z_1, self.G_test_2_)

        lossL2_1 = tf.add_n([ tf.nn.l2_loss(v) for v in self.G_vars_1 if 'W' in v.name ]) * 0.0
        lossL2_1_b = tf.add_n([ tf.nn.l2_loss(v) for v in self.G_vars_1 if 'B' in v.name ]) * 0.0
        lossL2_2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.G_vars_2 if 'W' in v.name ]) * 0.0
        lossL2_2_b = tf.add_n([ tf.nn.l2_loss(v) for v in self.G_vars_2 if 'B' in v.name ]) * 0.0

        # loss for G test
        self.invisibility_test = tf.reduce_mean(tf.multiply(tf.subtract(self.G_test_1, self.x_1), tf.subtract(self.G_test_1, self.x_1)))
        self.robustness_test = tf.reduce_mean(tf.abs(tf.subtract(self.G_test_2_, self.z_1)))

        self.G1_loss_total_test = (self.lambda1*self.invisibility_test) + (self.lambda2*self.robustness_test) + lossL2_1 + lossL2_1_b
        self.G2_loss_total_test = self.lambda3*self.robustness_test + lossL2_2 + lossL2_2_b

    def train(self, dataset_name, epoch, seed):
        # dataset load
        #data = h5py.File('01_Hologram_h5/train_data/20210301_global_std_train', 'r')
        #Host_train = np.asarray(data['data']).reshape([-1, self.height, self.width, 1])
        #condition_train = np.asarray(data['condition']).reshape([-1, 1, 5])
        #data2 = h5py.File('01_Hologram_h5/train_data/20210301_global_std_validation', 'r')
        #Host_test = np.asarray(data2['data']).reshape([-1, self.height, self.width, 1])
        #condition_test = np.asarray(data2['condition']).reshape([-1, 1, 5])
        # data = open('./train.json')
        # data = json.load(data)
        # Host_train = np.asarray(data['ampli']).reshape([-1, self.height, self.width, 1])
        # Host_train = Host_train * 2.0 - 1.0
        # Host_train_phs = np.asarray(data['phase']).reshape([-1, self.height, self.width, 1])
        # condition_train = np.asarray(data['condition']).reshape([-1, 1, 5])
        #
        # data2 = open('.//validation.json')
        # data2 = json.load(data2)
        # Host_test = np.asarray(data2['ampli']).reshape([-1, self.height, self.width, 1])
        # Host_test_phs = np.asarray(data2['phase']).reshape([-1, self.height, self.width, 1])
        # condition_test = np.asarray(data2['condition']).reshape([-1, 1, 5])

        data = h5py.File('/home/ipsl/hologram/20210301_global_std_train_amp', 'r')
        Host_train = np.asarray(data['data']).reshape([-1, self.height, self.width, 1])
        condition_train = np.asarray(data['condition']).reshape([-1, 1, 5])
        data2 = h5py.File('/home/ipsl/hologram/20210301_global_std_validation_amp', 'r')
        Host_test = np.asarray(data2['data']).reshape([-1, self.height, self.width, 1])
        condition_test = np.asarray(data2['condition']).reshape([-1, 1, 5])

        data0 = h5py.File('/home/ipsl/hologram/20210301_global_std_train_phase', 'r')
        Host_train_phs = np.asarray(data0['data']).reshape([-1, self.height, self.width, 1])
        data20 = h5py.File('/home/ipsl/hologram/20210301_global_std_validation_phase', 'r')
        Host_test_phs = np.asarray(data20['data']).reshape([-1, self.height, self.width, 1])



        num_train_data = len(Host_train)
        num_batches = num_train_data // self.batch_size

        print(Host_train.shape)
        print(Host_test.shape)

        self.build_model()

        # session creation
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))#config=tf.ConfigProto(device_count={'GPU':0}))
        self.sess.run(tf.global_variables_initializer())
        #vars_ = tf.all_variables()[-8:]
        #print(vars_)
        #init_new_vars_op = tf.initialize_variables(vars_)

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=1000)

        # summary writer
        if not os.path.exists('./log'):
            os.makedirs('./log')
        writer = tf.summary.FileWriter('./log/'+dataset_name+'.txt', self.sess.graph)

        # check-point restore if it exists
        load_flag, load_count = self.load_checkpoint('./ckt', dataset_name)
        if (load_flag):
            start_epoch = load_count//num_batches
            #start_batch_id = load_count - start_epoch * num_batches
            counter = load_count
        else:
            start_epoch = 0
            #start_batch_id = 0
            counter = 0
            
        # directory check
        directory = 'ckt/' + dataset_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = 'ckt_best/' + dataset_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = 'result/' + dataset_name+'/WMed/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = 'result/' + dataset_name+'/WM/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        

        index = np.arange(Host_train.shape[0])
        index_ = np.arange(Host_test.shape[0])
        text_list=[]

        NC_best = 0
        PSNR_best = 0

        recon_loss_train_gf = []
        recon_loss_test_gf = []
        for ep in range(start_epoch, epoch):
            
            start_time = time.time()

            G1_loss_total_ = []
            G2_loss_total_ = []
            PSNR_WMed_=[]
            PSNR_WMed_t_=[]
            NC_WM_=[]
            NC_WM_A1_=[]
            NC_WM_A2_=[]
            NC_WM_A22_=[]
            NC_WM_A3_=[]
            NC_WM_A4_=[]
            NC_WM_A5_=[]
            NC_WM_A6_=[]
            NC_WM_A8_=[]
            NC_WM_A13_=[]
            NC_WM_A14_=[]
            NC_WM_A15_=[]
            NC_WM_t_=[]
            NC_WM_t_A1_=[]
            NC_WM_t_A2_=[]
            NC_WM_t_A22_=[]
            NC_WM_t_A3_=[]
            NC_WM_t_A4_=[]
            NC_WM_t_A5_=[]
            NC_WM_t_A6_=[]
            NC_WM_t_A8_=[]
            NC_WM_t_A13_=[]
            NC_WM_t_A14_=[]
            NC_WM_t_A15_=[]
            D_real_1_=[]
            D_fake_1_=[]
            G1_loss_=[]
            invisibility_=[]
            robustness_=[]
            invisibility_t=[]
            robustness_t=[]
            WMed = []
            

            np.random.shuffle(index)
            Host_train = Host_train[index]
            Host_train_phs = Host_train_phs[index]
            condition_train = condition_train[index]
            
            np.random.shuffle(index_)
            Host_test = Host_test[index_]
            Host_test_phs = Host_test_phs[index_]
            condition_test = condition_test[index_]


            for bid in range(num_batches):
                #self.sess.run(init_new_vars_op)
                idx = np.random.choice(index, self.batch_size)
                Host_batch = Host_train[idx]
                Host_batch_phs = Host_train_phs[idx]
                condition_batch = condition_train[idx]

                WM_data_a = np.random.uniform(0, 1, [self.batch_size, self.height//16, self.width//16, 1])
                WM_data_a = np.round(WM_data_a)
                WM_data_b = WM_data_a-1
                WM_data = WM_data_a + WM_data_b

                for_recon_ori = np.asarray(Host_batch.reshape([self.batch_size, self.height, self.width]))
                for_recon_ori_phs = np.asarray(Host_batch_phs.reshape([self.batch_size, self.height, self.width]))

                #print('ori_source')
                #print(for_recon_ori[0])
                #print(for_recon_ori[0].shape)
                
                for_recon_fake = self.sess.run([self.G_fake_1], feed_dict={self.x_1 : Host_batch.astype(np.float32), self.z_1 : WM_data.astype(np.float32)})

                for_recon_fake = np.asarray(for_recon_fake).reshape([self.batch_size, self.height, self.width])
                #print('fake_source')
                #print(for_recon_fake[0])
                #print(for_recon_fake[0].shape)                

                recon_ori_ = []
                recon_fake_ = []
                recon_loss_list = []
                for i in range(self.batch_size):
                    if condition_batch[i][0, 0] == 0:  # ASM
                        if condition_batch[i][0, 1] == 0:  # Real
                            for_recon_ori_ = for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_batch[i][0, 2],condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_batch[i][0, 2],condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake - recon_ori) * (recon_fake - recon_ori)))

                        elif condition_batch[i][0, 1] == 1:  # Imag
                            for_recon_ori_ = 0 + for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = 0 + for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_batch[i][0, 2],
                                                                condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_batch[i][0, 2],
                                                                 condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake - recon_ori) * (recon_fake - recon_ori)))

                    elif condition_batch[i][0, 0] == 1:  # FDM
                        if condition_batch[i][0, 1] == 0:  # Real
                            for_recon_ori_ = for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_batch[i][0, 2],condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_batch[i][0, 2],condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake - recon_ori) * (recon_fake - recon_ori)))

                        elif condition_batch[i][0, 1] == 1:  # Imag
                            for_recon_ori_ = 0 + for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = 0 + for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_batch[i][0, 2], condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_batch[i][0, 2],  condition_batch[i][0, 3], condition_batch[i][0, 4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake - recon_ori) * (recon_fake - recon_ori)))


                recon_loss_list = np.asarray(recon_loss_list)
                 
                np.nan_to_num(recon_loss_list, copy=False)
                recon_ori_ = np.asarray(recon_ori_)            
                recon_fake_ = np.asarray(recon_fake_)                
                recon_loss_ = np.mean(recon_loss_list)
                #print(recon_loss_)

                recon_loss_train_gf.append(recon_loss_)

                # update generator network
                _, G1_loss_total, invisibility, PSNR_WMed = self.sess.run([self.G_optim_1_, self.G1_loss_total_, self.invisibility, self.psnr_test], 
                    feed_dict={
                    self.x_1 : Host_batch.astype(np.float32),
                    self.z_1 : WM_data.astype(np.float32),
                    self.recon_loss : recon_loss_.astype(np.float32)
                })
                # update generator network
                _, G2_result, G2_loss_total, robustness, \
                NC_WM, NC_WM_A1, NC_WM_A2, NC_WM_A22, NC_WM_A3, NC_WM_A4, NC_WM_A5, \
                NC_WM_A8, NC_WM_A13, \
                NC_WM_A14, NC_WM_A15 \
                    = self.sess.run([self.G_optim_2_, self.G_test_2_, self.G2_loss_total_, self.robustness_, \
                                     self.NC_WM_t_, self.NC_WM_t_A1, self.NC_WM_t_A2, self.NC_WM_t_A22, self.NC_WM_t_A3, self.NC_WM_t_A4, self.NC_WM_t_A5, \
                                     self.NC_WM_t_A8, self.NC_WM_t_A13, \
                                     self.NC_WM_t_A14, self.NC_WM_t_A15], 
                    feed_dict={
                    self.x_1 : Host_batch.astype(np.float32),
                    self.z_1 : WM_data.astype(np.float32),
                    self.recon_loss : recon_loss_.astype(np.float32)
                })
                #writer.add_summary(summary_str_2)

                invisibility_.append(invisibility)
                robustness_.append(robustness)

                G1_loss_total_.append(G1_loss_total)
                G2_loss_total_.append(G2_loss_total)

                PSNR_WMed_.append(PSNR_WMed)

                NC_WM_.append(NC_WM)
                NC_WM_A1_.append(NC_WM_A1)
                NC_WM_A2_.append(NC_WM_A2)
                NC_WM_A22_.append(NC_WM_A22)
                NC_WM_A3_.append(NC_WM_A3)
                NC_WM_A4_.append(NC_WM_A4)
                NC_WM_A5_.append(NC_WM_A5)
                NC_WM_A8_.append(NC_WM_A8)
                NC_WM_A13_.append(NC_WM_A13)
                NC_WM_A14_.append(NC_WM_A14)
                NC_WM_A15_.append(NC_WM_A15)

                
                WM_data_ = np.round(np.clip((WM_data +1)/2, 0, 1))
                counter += 1
    
            # test generator network
            for bid_t in range(round(Host_test.shape[0]/self.batch_size)) :

                
                Host_test_batch = Host_test[self.batch_size*bid_t:self.batch_size*(bid_t+1)]
                Host_test_batch_phs = Host_test_phs[self.batch_size*bid_t:self.batch_size*(bid_t+1)]

                condition_test_batch = condition_test[self.batch_size*bid_t:self.batch_size*(bid_t+1)]

                for_recon_ori = np.asarray(Host_test_batch.reshape([self.batch_size, self.height, self.width]))
                for_recon_ori_phs = np.asarray(Host_test_batch_phs.reshape([self.batch_size, self.height, self.width]))
                for_recon_fake = self.sess.run([self.G_test_1], feed_dict={self.x_1 : Host_test_batch.astype(np.float32), self.z_1 : WM_data.astype(np.float32)})
                for_recon_fake = np.asarray(for_recon_fake).reshape([self.batch_size, self.height, self.width])
                
                recon_ori_ = []
                recon_fake_ = []
                recon_loss_list = []

                for i in range(self.batch_size):
                    if condition_test_batch[i][0,0] == 0: # ASM
                        if condition_test_batch[i][0,1] == 0: # Real
                            for_recon_ori_ = for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake-recon_ori)*(recon_fake-recon_ori)))
                        elif condition_test_batch[i][0,1] == 1: # Imag
                            for_recon_ori_ = 0 + for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = 0 + for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake-recon_ori)*(recon_fake-recon_ori)))

                    elif condition_test_batch[i][0,0] == 1: # FDM
                        if condition_test_batch[i][0,1] == 0: # Real
                            for_recon_ori_ = for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake-recon_ori)*(recon_fake-recon_ori)))

                        elif condition_test_batch[i][0,1] == 1: # Imag
                            for_recon_ori_ = 0 + for_recon_ori[i] * np.exp(1j * for_recon_ori_phs[i])
                            for_recon_fake_ = 0 + for_recon_fake[i] * np.exp(1j * for_recon_ori_phs[i])
                            recon_ori = self.ASM_reconstruction(for_recon_ori_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_fake = self.ASM_reconstruction(for_recon_fake_, condition_test_batch[i][0,2], condition_test_batch[i][0,3], condition_test_batch[i][0,4])
                            recon_ori_.append(recon_ori)
                            recon_fake_.append(recon_fake)
                            recon_loss_list.append(np.mean((recon_fake-recon_ori)*(recon_fake-recon_ori)))

                recon_loss_list = np.asarray(recon_loss_list)                
                np.nan_to_num(recon_loss_list, copy=False)
                recon_ori_ = np.asarray(recon_ori_)            
                recon_fake_ = np.asarray(recon_fake_)
                recon_loss_ = np.mean(recon_loss_list)

                recon_loss_test_gf.append(recon_loss_)

                WMed_, g_test_2, PSNR_WMed_t, \
                invisibility_test, robustness_test, \
                NC_WM_t, NC_WM_t_A1, NC_WM_t_A2, NC_WM_t_A22, NC_WM_t_A3, NC_WM_t_A4, NC_WM_t_A5, \
                NC_WM_t_A8, NC_WM_t_A13, \
                NC_WM_t_A14, NC_WM_t_A15 \
                    = self.sess.run([self.G_test_1, self.G_test_2_, self.psnr_test, \
                                     self.invisibility_test, self.robustness_test, \
                                     self.NC_WM_t_, self.NC_WM_t_A1, self.NC_WM_t_A2, self.NC_WM_t_A22, self.NC_WM_t_A3, self.NC_WM_t_A4, self.NC_WM_t_A5, \
                                     self.NC_WM_t_A8, self.NC_WM_t_A13, \
                                     self.NC_WM_t_A14, self.NC_WM_t_A15], 
                    feed_dict={
                    self.x_1 : Host_test_batch.astype(np.float32),
                    self.z_1 : WM_data.astype(np.float32)
                })
    
                PSNR_WMed_t_.append(PSNR_WMed_t)
                invisibility_t.append(invisibility_test)
                robustness_t.append(robustness_test)
    
                NC_WM_t_.append(NC_WM_t) 
                NC_WM_t_A1_.append(NC_WM_t_A1) 
                NC_WM_t_A2_.append(NC_WM_t_A2) 
                NC_WM_t_A22_.append(NC_WM_t_A22) 
                NC_WM_t_A3_.append(NC_WM_t_A3) 
                NC_WM_t_A4_.append(NC_WM_t_A4) 
                NC_WM_t_A5_.append(NC_WM_t_A5) 
                NC_WM_t_A8_.append(NC_WM_t_A8) 
                NC_WM_t_A13_.append(NC_WM_t_A13) 
                NC_WM_t_A14_.append(NC_WM_t_A14) 
                NC_WM_t_A15_.append(NC_WM_t_A15) 
                WMed.append(WMed_)

            Host_test_ = np.clip((Host_test+1)/2, 0, 1)
            WMed = np.asarray(WMed).reshape([-1, self.height, self.width, 1])
            WMed = np.clip((WMed+1)/2, 0, 1)
            g_test_2 = np.round(np.clip((g_test_2 +1)/2, 0, 1))
    
            # log print
            print('Epoch : [%2d], time : %4.4fs, g1_loss:%.8f[inv:%.4f/rob:%.4f], g1_loss_test:%.8f[inv:%.4f/rob:%.4f], \
                   PSNR_WMed:%.2f, PSNR_WMed_test:%.2f, g2_loss:%.8f, g2_loss_test:%.8f,\
                   NC_WM_train:[%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f], \
                   NC_WM_test :[%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f]' 
                % (ep, time.time()-start_time, np.mean(G1_loss_total_), np.mean(invisibility_)*self.lambda1, np.mean(robustness_)*self.lambda2, \
                    np.mean(invisibility_t)*self.lambda1+np.mean(robustness_t)*self.lambda2, \
                    np.mean(invisibility_t)*self.lambda1, np.mean(robustness_t)*self.lambda2, \
                    np.mean(PSNR_WMed_), np.mean(PSNR_WMed_t_), np.mean(G2_loss_total_), np.mean(robustness_t)*self.lambda3, \
                    np.mean(NC_WM_), np.mean(NC_WM_A1_), np.mean(NC_WM_A2_), np.mean(NC_WM_A22_), np.mean(NC_WM_A3_), np.mean(NC_WM_A4_), np.mean(NC_WM_A5_), \
                    np.mean(NC_WM_A8_), np.mean(NC_WM_A13_), \
                    np.mean(NC_WM_A14_), np.mean(NC_WM_A15_), \
                    np.mean(NC_WM_t_), np.mean(NC_WM_t_A1_), np.mean(NC_WM_t_A2_), np.mean(NC_WM_t_A22_), np.mean(NC_WM_t_A3_), np.mean(NC_WM_t_A4_), np.mean(NC_WM_t_A5_), \
                    np.mean(NC_WM_t_A8_), np.mean(NC_WM_t_A13_), \
                    np.mean(NC_WM_t_A14_), np.mean(NC_WM_t_A15_)))

            text_list.append('Epoch : [%2d], time : %4.4fs, g1_loss:%.8f[inv:%.4f/rob:%.4f], g1_loss_test:%.8f[inv:%.4f/rob:%.4f], PSNR_WMed:%.2f, PSNR_WMed_test:%.2f, g2_loss:%.8f, g2_loss_test:%.8f, NC_WM_train:[%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f], NC_WM_test :[%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f]' 
                % (ep, time.time()-start_time, np.mean(G1_loss_total_), np.mean(invisibility_)*self.lambda1, np.mean(robustness_)*self.lambda2, \
                    np.mean(invisibility_t)*self.lambda1+np.mean(robustness_t)*self.lambda2, \
                    np.mean(invisibility_t)*self.lambda1, np.mean(robustness_t)*self.lambda2, \
                    np.mean(PSNR_WMed_), np.mean(PSNR_WMed_t_), np.mean(G2_loss_total_), np.mean(robustness_t)*self.lambda3, \
                    np.mean(NC_WM_), np.mean(NC_WM_A1_), np.mean(NC_WM_A2_), np.mean(NC_WM_A22_), np.mean(NC_WM_A3_), np.mean(NC_WM_A4_), np.mean(NC_WM_A5_), \
                    np.mean(NC_WM_A8_), np.mean(NC_WM_A13_), \
                    np.mean(NC_WM_A14_), np.mean(NC_WM_A15_), \
                    np.mean(NC_WM_t_), np.mean(NC_WM_t_A1_), np.mean(NC_WM_t_A2_), np.mean(NC_WM_t_A22_), np.mean(NC_WM_t_A3_), np.mean(NC_WM_t_A4_), np.mean(NC_WM_t_A5_), \
                    np.mean(NC_WM_t_A8_), np.mean(NC_WM_t_A13_), \
                    np.mean(NC_WM_t_A14_), np.mean(NC_WM_t_A15_)))
            # model saver
            self.save_checkpoint('./ckt', dataset_name, counter)

            if (NC_best < np.mean(NC_WM_t_)):
                NC_best = np.mean(NC_WM_t_)
                self.save_checkpoint('./ckt_best', dataset_name, counter)
            if (PSNR_best < np.mean(PSNR_WMed_t_)):
                PSNR_best = np.mean(PSNR_WMed_t_)
                self.save_checkpoint('./ckt_best', dataset_name, counter)

            # visualize 
            n = np.random.choice(self.batch_size)
            WMed = WMed.reshape(-1, self.height,self.width)
            Host_test_ = Host_test_.reshape(-1, self.height,self.width)
            g_test_2 = g_test_2.reshape(-1, self.height//16, self.width//16)
            WM_data_ = WM_data_.reshape(-1, self.height//16, self.width//16)
            scipy.misc.toimage(WMed[n], cmin=0.0, cmax=1.0).save('result/'+dataset_name+'/WMed/'+str(ep)+'_'+str(bid)+'_fake.png')
            scipy.misc.toimage(Host_test_[n], cmin=0.0, cmax=1.0).save('result/'+dataset_name+'/WMed/'+str(ep)+'_'+str(bid)+'_real.png')
            scipy.misc.toimage(g_test_2[n], cmin=0.0, cmax=1.0).save('result/'+dataset_name+'/WM/'+str(ep)+'_'+str(bid)+'_fake.png')
            scipy.misc.toimage(WM_data_[n], cmin=0.0, cmax=1.0).save('result/'+dataset_name+'/WM/'+str(ep)+'_'+str(bid)+'_real.png')
            
            if ((ep%10)==0):
                f = open('result/'+dataset_name+'/result.txt', 'a')
                f.write('\n')
                f.writelines('\n'.join(text_list))
                f.close()
                text_list=[]        
                np.save('result/'+dataset_name+'/recon_loss_train', recon_loss_train_gf)
                np.save('result/'+dataset_name+'/recon_loss_test', recon_loss_test_gf)
        f = open('result/'+dataset_name+'/result.txt', 'a')
        f.write('\n')
        f.writelines('\n'.join(text_list))
        f.close()
        text_list=[]

    def save_checkpoint(self, checkpoint_dir, dataset_name, step):
        checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, dataset_name + '.model'), global_step=step)

    def load_checkpoint(self, checkpoint_dir, dataset_name):
        import re

        checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print( "로드" ,checkpoint_dir,ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

if __name__ == '__main__' :
    #250
    width = 128
    height = 128

    path = "v9_holo_amplitude"
    print(path)
    model = WM_NN(width=width, height=height, kernel1=3, kernel2=3, batch_size=80,
                  d1_lr=0.0001, g1_lr=0.0001, gan2_lr=0.00001, beta1=0.5, beta2=0.999,
                  lambda1=45.0, lambda2=0.2, lambda3=20)
    model.train(path, epoch=4000, seed=None)
