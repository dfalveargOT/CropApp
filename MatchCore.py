#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:45:11 2019
Copyright © 2019 DataRock S.A.S. All rights reserved.
@author: DavidFelipe
Second Module
Matching each object with the corresponding coordinates
"""
try:
    import numpy as np
    import cv2
    import time
    import progressbar
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED")


print(" %% SECOND MODULE %%")
print(" -- Matching object module for images -- Check the current progress --")


class MatchCore:
    """
    Procedure to find the objects and extract window information 
    of the current object in process
    """
    def __init__(self, imageRGB, Rlayer1, Rlayer2, Rlayer3, Vmean, Vrange):
        print(" ")
        print("MatchCore process")
        self.image = imageRGB
        self.layer1 = Rlayer1
        self.layer2 = Rlayer2
        self.layer3 = Rlayer3
        self.vector_mean = Vmean
        self.vector_range = Vrange
        
    def CoreFinding(self, layer):
        
        shapeLayer = np.array(layer.shape)
        val = shapeLayer.shape
        print(" ")
        print(" matchCore proceesing corefinding ")
        if(val[0] == 3):
#            print("RGB format found")
            ## Processing layer 1
            widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
            bar = progressbar.ProgressBar(widgets=widgets, maxval=5)
            bar.start()
            Mlayer1, mask_layer1, vector_layer1, count_layer1, blob_layer1 = self.ObjectMatch(self.image, layer[:,:,0])
            Mlayer2, mask_layer2, vector_layer2, count_layer2, blob_layer2 = self.ObjectMatch(self.image, layer[:,:,1])
            Mlayer3, mask_layer3, vector_layer3, count_layer3, blob_layer3 = self.ObjectMatch(self.image, layer[:,:,2])
            bar.update(1)
            mask_layer = np.zeros_like(layer)
            blob_layer = np.zeros_like(layer)
            bar.update(2)
            blob_layer[:,:,0] = blob_layer1
            blob_layer[:,:,1] = blob_layer2
            blob_layer[:,:,2] = blob_layer3
            bar.update(3)
            mask_layer[:,:,0] = mask_layer1
            mask_layer[:,:,1] = mask_layer2
            mask_layer[:,:,2] = mask_layer3
            bar.update(4)
            count_layer = np.array([count_layer1, count_layer2, count_layer3])
            vector_layer = [vector_layer1, vector_layer2, vector_layer3]
            bar.update(5)
            return Mlayer1, Mlayer2, Mlayer3, mask_layer, vector_layer, blob_layer
            
        elif(val[0] == 2):
#            print("Fourier result processing")
            match_layer, mask_layer, vector_layer, count_layer, blob_layer = self.ObjectMatch(self.image, layer)
            
            return match_layer, mask_layer, vector_layer, blob_layer
        else:
            print(" PROBLEM WITH THE DATA, THE SIZE OF THE ENTRY IS NOT CORRECT, VERIFY THE LAST CONVERTIONS")
    
#    cv2.CHAIN_APPROX_TC89_L1
                
    def ObjectMatch(self, image, mask):
        x, y, _ = image.shape
        image_mark = np.copy(self.image)
        object_mask = np.zeros([x,y])
        contours,hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        vector_object = np.array([0,0,0,0])
        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= 7) and (
                h >= 7) and (w <= 500) and (h <= 500)
            if not contour_valid:
                continue
            counter += 1
            ## Definimos bounding box para el tracker
            boundingBox = np.array([x,y,w,h])
            ## Definimos tipo de Tracker
            # getting center of the bounding box
            x1 = int(w / 2)
            y1 = int(h / 2)
            cx = x + x1
            cy = y + y1
            cv2.circle(image_mark, (cx, cy), 10, (255, 255, 255), 2)
            cv2.circle(image_mark, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(object_mask, (cx, cy), 10, (255, 255, 255), 4)
            ## Vector of the current object
            vector_object = np.vstack((vector_object, boundingBox))
        
        ###3 Detection using Blob
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(self.image)
        image_blob = mask.copy()
#        print(keypoints.type)
        for kp in keypoints:
            image_blob = cv2.drawMarker(image_blob, tuple(int(i) for i in kp.pt), color=(0,0,255))
            
        return image_mark, object_mask, vector_object, counter, image_blob

