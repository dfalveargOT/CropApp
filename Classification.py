#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:25:16 2019
Copyright © 2019 DataRock S.A.S. All rights reserved.
@author: DavidFelipe
Machine learning phase
"""


try:
    import numpy as np
    import time
    import progressbar
    import cv2
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED")


print(" %% FOURTH MODULE %%")
print(" -- Machine learining process  -- Check the current progress --")


class Classification:
    """
    Procedure to process the coordinates and the geometric properties of each object
    segmented and found
    """
    def __init__(self, imageRGB):
        self.image = imageRGB
        self.x, self.y, self.z = imageRGB.shape
        
    def mergeImages(self, image1, image2, image3):
        image5 = np.zeros_like(image1)
        merge_sup = np.concatenate((image1, image2), axis=1)
        merge_inf = np.concatenate((image5, image3), axis=1)
        merge = np.concatenate((merge_sup, merge_inf), axis=0)
        return merge
            
    
    def windowObject(self, vectorList, meanArea, pathGeneral, image_Matched, image_mask, id_vector, layer):
        path = pathGeneral + "L" + str(layer) + str(id_vector) + "/"
        counter_A = 0
        counter_F = 0
        counter_D = 0
        counter_Q = 0
        if(type(vectorList) == list):
            vector = vectorList[id_vector]
            meanAr = meanArea[id_vector]
            x,y = vector.shape
        else:
            x,y = vector.shape
        for obj in range (0,x):
            line = vector[obj]
            cx = int(line[1])
            cy = int(line[2])
            dim_1 = int(((np.sqrt(line[3])))*1.1)
            dim_2 = int(((np.sqrt(line[3])))*2)
            dim = int(((np.sqrt(meanAr))))
            if(cx > 0 and cy > 0):
                l1 = cx - dim
                l2 = cy - dim
                d1 = cx - dim_1
                d2 = cy - dim_1
                if(d1 > self.x):
                    d1 = self.x
                if(d2 > self.y):
                    d2 = self.y
                if(l1 > self.x or cx > self.x):
                    l1 = self.x
                    cx = self.x
                if(l2 > self.y or cy > self.y):
                    l2 = self.y
                    cy = self.y
                if(l1 <= 0 or cx <= 0):
                    l1 = 0
                if(d1<=0):
                    d1 = 0
                if(l2 < 0 or cy <= 0):
                    l2 = 0
                if(d2<=0):
                    d2 = 0
                frac_matched = image_Matched[l2:cy+dim,l1:cx+dim,:]
                frac_matched = cv2.resize(frac_matched, (100,100))
                frac_image = self.image[l2:cy+dim,l1:cx+dim,:]
                frac_image = cv2.resize(frac_image, (100,100))
                frac_mask = 255*image_mask[l2:cy+dim,l1:cx+dim]
                frac_mask = cv2.resize(frac_mask, (100,100))
                image_merged = self.mergeImages(frac_image, frac_matched, frac_mask)
                
                frac_matched_1 = image_Matched[l2:cy+dim,l1:cx+dim,:]
                frac_matched_1 = cv2.resize(frac_matched_1, (100,100))
                frac_image_1 = self.image[l2:cy+dim,l1:cx+dim,:]
                frac_image_1 = cv2.resize(frac_image, (100,100))
                frac_mask_1 = 255*image_mask[l2:cy+dim,l1:cx+dim]
                frac_mask_1 = cv2.resize(frac_mask, (100,100))
                image_merged_1 = self.mergeImages(frac_image_1, frac_matched_1, frac_mask_1)
                
                frac_matched_2 = image_Matched[l2:cy+dim,l1:cx+dim,:]
                frac_matched_2 = cv2.resize(frac_matched_1, (100,100))
                frac_image_2 = self.image[l2:cy+dim,l1:cx+dim,:]
                frac_image_2 = cv2.resize(frac_image, (100,100))
                frac_mask_2 = 255*image_mask[l2:cy+dim,l1:cx+dim]
                frac_mask_2 = cv2.resize(frac_mask, (100,100))
                image_merged_2 = self.mergeImages(frac_image_2, frac_matched_2, frac_mask_2)
                
                image = np.concatenate((image_merged, image_merged_1, image_merged_2), axis=1)
                
                try:  
                    cv2.putText(image, "Aceptados : " + str(counter_A), (5, 110) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(image, "Total : "+str(x), (5, 130),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(image, "MediaArea : "+str(round(meanAr, 1)) + " Obj " + str(round(np.sqrt(line[3]),1)), (5, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.imshow("Image", image)
                except:
                    print(image_merged.shape)
                
                key = cv2.waitKey(0)
                if(key == ord("a")):
                    "Save as right object"
                    cv2.imwrite(path + "A"+ str(counter_A) +".jpg", frac_image.astype('uint8'))
                    counter_A += 1

                if(key==ord("s")):
                    "Save as wrong object"
                    cv2.imwrite(path + "F"+ str(counter_F) +".jpg", frac_image.astype('uint8'))
                    counter_F += 1

                if(key==ord("d")):
                    "Save as double object"
                    cv2.imwrite(path + "D"+ str(counter_F) +".jpg", frac_image.astype('uint8'))
                    counter_D += 1
                
                if(key==ord("z")):
                    "Save as double object"
                    counter_Q += 1

                if(key==ord("q")):  
                    cv2.destroyAllWindows()
                    print(counter_A)
                    print(counter_F)
                    print(counter_D)
                    print(counter_Q)
                    break
                cv2.destroyAllWindows()
                

            
            