#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:18:24 2019
Copyright © 2019 DataRock S.A.S. All rights reserved.
@author: DavidFelipe

Script de preprocesamiento de imagenes

"""
try:
    import numpy as np
    from skimage.color import rgb2yiq, rgb2lab, rgb2ycbcr, rgb2gray
    import cv2
    from skimage.morphology import disk
    from skimage.morphology import dilation
    from skimage.morphology import erosion
    import time
    import progressbar
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED")


print(" %% FIRST MODULE %%")
print(" -- Preprocesing module for images -- See the current progress --")

class preprocessing:
    """ Module one for preprocessing images, software will implement some image transformations """
    def __init__(self, percentVal, percentFourier, imageRGB):
        """
        The entry image come as numpy format
        """
        print("Preprocessing - Layer processing")
        self.percent = percentVal
        self.image = cv2.bilateralFilter(imageRGB,20,150,150)
        x, y, z = imageRGB.shape
        self.zero = np.zeros([x,y])
        self.yiq = self.zero
        self.lab = self.zero
        self.ycb = self.zero
        self.hls = self.zero
        self.yuv = self.zero
        self.Array_layer1 = np.zeros([x,y,z], 'uint8')
        self.Array_layer2 = np.zeros([x,y,z], 'uint8')
        self.percentFou = percentFourier
        
    def processingLayer1(self):
        """
        Processing for YIQ, LAB, YCBC color models
        """
        print(" ")
        print("Preprocessing - Layer 1 processing")
        widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=3)
        bar.start()
        self.yiq = rgb2yiq(self.image) ## 1
        self.lab = rgb2lab(self.image) ## 2
        self.ycb = rgb2ycbcr(self.image) ## 2
        bar.update(1)
        ## morph procedure 
        binary_yiq = self.morfprocess(self.yiq, 2, 1, 1, 0, self.percent)
        binary_lab = self.morfprocess(self.lab, 2, 1, 0, 0, self.percent)
        binary_ycb = self.morfprocess(self.ycb, 2, 1, 0, 0, self.percent)
        bar.update(2)
        ## Arragne the information
        self.Array_layer1[:,:,0] = binary_yiq
        self.Array_layer1[:,:,1] = binary_lab
        self.Array_layer1[:,:,2] = binary_ycb
        bar.update(3)
        ## Return the information
        return self.Array_layer1
        
    def processingLayer2(self):
        """
        Processing for HLS, LUV, YUV color models
        """
        print(" ")
        print("Preprocessing - Layer 2 processing")
        widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=3)
        bar.start()
        
        ## CONVERSIÓN CVCOLOR HLS, LUV, YUV
        self.hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS_FULL) ## 0
        self.luv = cv2.cvtColor(self.image, cv2.COLOR_RGB2LUV) ## 2
        self.yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV) ## 1
        bar.update(1)
        ## morph procedure 
        binary_hls = self.morfprocess(self.hls, 0, 1, 1, 0, self.percent)
        binary_luv = self.morfprocess(self.luv, 2, 1, 0, 0, self.percent)
        binary_yuv = self.morfprocess(self.yuv, 1, 1, 0, 0, self.percent)
        bar.update(2)
        ## Arragne the information
        self.Array_layer2[:,:,0] = binary_hls
        self.Array_layer2[:,:,1] = binary_luv
        self.Array_layer2[:,:,2] = binary_yuv
        bar.update(3)
        ## Return the information
        return self.Array_layer2
    
    def processingLayer3(self):
        """
        Processing for RGB image using Fourier transform to aproach the segmentation
        """
        print(" ")
        print("Preprocessing - Layer 3 processing")
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = (20*np.log(np.abs(fshift))).astype('uint8')
        
        ## Mean procedure
        mean_fourier = np.array([magnitude_spectrum[:,:,0].mean(), magnitude_spectrum[:,:,1].mean(), magnitude_spectrum[:,:,2].mean()])
        
        ## Range procedure
        hist_01= cv2.calcHist( [magnitude_spectrum], [0, 1], None, [256, 256], [0, 256, 0, 256] )
        hist_02= cv2.calcHist( [magnitude_spectrum], [0, 2], None, [256, 256], [0, 256, 0, 256] )
        val_01 = np.percentile(hist_01, self.percentFou)
        val_02 = np.percentile(hist_02, self.percentFou)
        mask_01 = hist_01 > val_01
        mask_02 = hist_02 > val_02
        cnts_01, _ = cv2.findContours((mask_01.copy()).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_02, _ = cv2.findContours((mask_02.copy()).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_01:
            (x1,y1,w1,h1) = cv2.boundingRect(np.array(c))
        for h in cnts_02:
            (x2,y2,w2,h2) = cv2.boundingRect(np.array(h))
            
        ### Range fourier
        rangeFourier = np.array([[x2,x2+w2],[x1,x1+w1],[y1,y1+h1]])
        
        ##### Procesing of the image uding mean aproach
        
        fourier = np.copy(magnitude_spectrum)
        image_trasnform = np.copy(self.image)
        size_x, size_y, size_c = fourier.shape
        binary_fourier = np.zeros([size_x, size_y], dtype='uint8')
        val = np.zeros(size_c)
        ### Utilizando media de color
        percent = 0.83
        inicio = time.time()
        area = size_x*size_y
        widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=area)
        bar.start()
        for x in range(0, size_x):
            for y in range(0, size_y):
                for c in range(0,size_c):
                    val[c] = np.abs(1-((image_trasnform[x,y,c])- mean_fourier[c])/255)
                    if(val[c] < percent):
                        val[0] = 0
                        break
                    elif(val[c] >= percent):
                        val[c] = 1 #se cumple la condición de rango se deja el pixel
                if(val[0]==0):
                    binary_fourier[x,y] = 0
                elif (val.all()==1):
                    binary_fourier[x,y] = 1
                val = np.zeros(size_c)
            fraction = x*y
            bar.update(fraction)
        final = time.time() - inicio
        bar.update(area)
        print("Tiempo de procesamiento : ", round(final,2), "Segundos")
        
        ### Morphological Process
        # First do a dilatation
        radio = 2
        sel = disk(radio)
        binary_dilat1 = dilation(binary_fourier, sel)
        for i in range(0,2):
            binary_dilat1 = dilation(binary_dilat1,sel)
        # Second erase little objects
        radio = 5
        sel = disk(radio)
        binary_erosion1 = erosion(binary_dilat1.copy(),sel)
        for i in range(0,2):
            binary_erosion1 = erosion(binary_erosion1,sel)
        # Then dilate again 
        radio = 3
        sel = disk(radio)
        binary = dilation(binary_erosion1.copy(),sel)
        
        return binary, binary_fourier, rangeFourier, mean_fourier
        
        
    def morfprocess(self, imagePros, valPosition, valRadio, valInvert, valDigitalize, valPercent):
        """
        Morphological processing of all information
        -imagePros : numpy image to process
        -valPosition : model color layer
        -valRadio : radio of morphological procedure
        -valInvert : if the convertión came in BW
        -valPercent : percent of the histogram to binarize
        """
        mask = rgb2gray(imagePros[:,:,valPosition])
        if(valInvert == 1):
            mask = 255 - mask
        val = np.percentile(mask,valPercent)
        binary = mask > val
        ## Erosion
        radio = 1
        selD = disk(radio)
        binary = (erosion(binary,selem=selD)).astype('uint8')
        
        ## Dilation
        radio = 3
        sel = disk(radio)
        binary = dilation(binary, sel)
        for i in range(0,1):
            binary = dilation(binary, sel)
            
        ## Erosion
        radio = 2
        sel = disk(radio)
        binary = erosion(binary, sel)
        for i in range(0,2):
            binary = erosion(binary, sel)
        ## Dilation
        radio = 2
        sel = disk(radio)
        binary = dilation(binary, sel)
        
        return binary


        
        
        