#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:45:36 2019
Copyright © 2019 DataRock S.A.S. All rights reserved.
@author: DavidFelipe
Select the objects that would be analized for each layer
"""

try:
    import numpy as np
    from operator import itemgetter
    import time
    import progressbar
    import cv2
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED")


print(" %% THIRD MODULE %%")
print(" -- Select the objects from the minimum  -- Check the current progress --")


class MarkProcess:
    """
    Procedure to process the coordinates and the geometric properties of each object
    segmented and found
    """
    def __init__(self, imageRGB, diameter_range, flagMinimize):
        print("MarkProcess Process")
        self.image = imageRGB
        self.diameter = diameter_range
        self.flag = flagMinimize
        self.diameterMark = 16
        self.markGate = 58
    
    def objectMatch(self, vectorL1, vectorL2, vectorL3):
        
        """
        Each vector is a list of three vectors containing the information
        of each sublayer of each layer
        """
        print(" ")
        print("MarkProcess object match process ")
        widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=3)
        bar.start()
        inicio = time.time()
        L1_organized = self.layer_organize(vectorL1)
        L2_organized = self.layer_organize(vectorL2)
        L3_organized = self.layer_organize(vectorL3)
        bar.update(1)
        L1_minimized =[]
        L2_minimized =[]
        L3_minimized =[]
        bar.update(2)
        if(self.flag == 1):
            ### Minimize the vector sparse
            L10_minimize = self.minimize(L1_organized[0], self.diameter)
            L11_minimize = self.minimize(L1_organized[1], self.diameter)
            L12_minimize = self.minimize(L1_organized[2], self.diameter)
            L20_minimize = self.minimize(L2_organized[0], self.diameter)
            L21_minimize = self.minimize(L2_organized[1], self.diameter)
            L22_minimize = self.minimize(L2_organized[2], self.diameter)
            L3_minimized = self.minimize(L3_organized, self.diameter)
            L1_minimized = [L10_minimize, L11_minimize, L12_minimize]
            L2_minimized = [L20_minimize, L21_minimize, L22_minimize]
            final = time.time() - inicio
            print(final)
            bar.update(3)
            print(" ")
            print("MarkProcess Ended with minimize function")
            
            return L1_minimized, L2_minimized, L3_minimized
        else:
            final = time.time() - inicio
            print(final)
            bar.update(3)
            print(" ")
            print("MarkProcess Ended - organized")
            
            return L1_organized, L2_organized, L3_organized
        
    
    def layer_organize(self, vector_layer):
        if(type(vector_layer) == list):
            vector1_organized = self.organize_vector(vector_layer[0])
            vector2_organized = self.organize_vector(vector_layer[1])
            vector3_organized = self.organize_vector(vector_layer[2])
            vector_organized = [vector1_organized, vector2_organized, vector3_organized]
        else:
            vector_organized = self.organize_vector(vector_layer)
        return vector_organized
            
        
    def organize_vector(self, vector):
        vector_metrics = np.array([0,0,0,0])
        x,y = vector.shape
        for i in range(1,x):
            line = vector[i]
            w = line[2]
            h = line[3]
            x1 = int(w / 2)
            y1 = int(h / 2)
            cx = int(line[0] + x1)
            cy = int(line[1] + y1)
            distance = (cx**2) + (cy**2)
            distance = int(np.sqrt(distance))
            area = int(w * h)
            line_block = np.array([distance, cx, cy, area])
            vector_metrics = np.vstack((vector_metrics, line_block))
        vector_sort = vector_metrics[vector_metrics[:,0].argsort()]
        return vector_sort
    
    def minimize(self, vector, diameter):
        """
        Function to compare and delete some points 
        """
        x, y = vector.shape
        widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
        bar2 = progressbar.ProgressBar(widgets=widgets, maxval=x)
        bar2.start()
        minimize_vector = np.array([0,0,0,0] )
        c = 1
        j = 2
        while(c <= x-2):
            line = vector[c]
            val = line[0]
            array = np.copy(line)
            counter = 0
            while(j <= x-1):
                lineCompare = vector[j]
#                print(lineCompare[0])
                difference = lineCompare[0] - val
                if(difference <= diameter):
                    array = np.vstack((array, lineCompare))
                    counter += 1
                elif(difference > diameter):
                    try:
                        distance_mean = np.mean(array[:,0])
                        cx_mean = np.average(array[:,1])
                        cy_mean = np.average(array[:,2])
                        area_mean = np.average(array[:,3])
                        result = np.array([distance_mean, cx_mean, cy_mean, area_mean])
                        c = c + counter
                    except:
                        result = array
                    break
                j += 1
            minimize_vector = np.vstack((minimize_vector, result))
            c += 1
            j = c + 1
#            print(c)
            bar2.update(c)
        bar2.update(x)
        return minimize_vector

    
    def imageMatch(self, vector):
        """
        Process to mark the coordinates in the current image process
        """
        print(" ")
        x,y = vector.shape
        widgets = [progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.ETA(),
                    ' ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=x)
        bar.start()
        mark_image = np.copy(self.image)
        for i in range(0, x):
            bar.update(i)
            line = vector[i]
            cx = int(line[1])
            cy = int(line[2])
            cv2.circle(mark_image, (cx, cy), 10, (255, 255, 255), 2)
            cv2.circle(mark_image, (cx, cy), 4, (0, 0, 255), -1)
        bar.update(x)
        return mark_image
    
    def meanGeometry(self, vector):
        if(type(vector) == list):
            vector1 = vector[0]
            vector2 = vector[1]
            vector3 = vector[2]
            vector1_mean = np.mean(vector1[:,3])
            vector2_mean = np.mean(vector2[:,3])
            vector3_mean = np.mean(vector3[:,3])
            vector_mean = np.array([vector1_mean, vector2_mean, vector3_mean])
        else:
            vector_mean = np.mean(vector[:,3])
        return vector_mean
    
    def imageGrouped(self, vector):
        """
        Process to mark the coordinates in the current image process
        """
        mask_image = np.zeros_like(self.image)
        mark_image = np.copy(self.image)
        x,y = vector.shape
        for i in range(0, x):
            line = vector[i]
            cx = int(line[1])
            cy = int(line[2])
            cv2.circle(mask_image, (cx, cy), self.diameterMark, (255, 255, 255), -1) #### Aqui hay un valor que puede cambiar dependiendo de la escala
            cv2.circle(mark_image, (cx, cy), 10, (255, 255, 255), 2)
            cv2.circle(mark_image, (cx, cy), 4, (0, 0, 255), -1)
        return mark_image, mask_image
    
    
    def Vagroup(self, vector1, vector2, vector3, vector4):
        grouped = np.vstack([vector1, vector2, vector3, vector4])
        grouped_organized = self.layer_organize(grouped)
        grouped_minimized = self.minimize(grouped_organized, 8)
        ## With the minimized function
        image_minimized = self.imageGrouped(grouped_minimized)
        ## Raw vector integrated
        image_grouped, mask_image = self.imageGrouped(grouped)
        
        grouped_vector = [grouped, grouped_organized, grouped_minimized]
        
        return image_grouped, mask_image, grouped_vector
    
    
    def countIntegration(self, image, mask_image):
        x, y, _ = image.shape
        image_mark = np.copy(self.image)
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        contours,hierachy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        vector_object = np.array([0,0,0,0])
        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= 5) and (
                h >= 5) and (w <= 500) and (h <= 500)
            if not contour_valid:
                continue
            
            ## Definimos bounding box para el tracker
            boundingBox = np.array([x,y,w,h])
            ## Definimos tipo de Tracker
            # getting center of the bounding box
            x1 = int(w / 2)
            y1 = int(h / 2)
            cx = x + x1
            cy = y + y1
            if(w > self.markGate or h > self.markGate):
                cv2.circle(image_mark, (cx, cy), 14, (255, 255, 255), 2)
                cv2.circle(image_mark, (cx, cy), 7, (255, 0, 0), -1)
                cv2.circle(mask_image, (cx, cy), 6, (0, 0, 255), -1)
                counter += 2
            elif(w > self.markGate and h > self.markGate):
                counter += 3
                cv2.circle(image_mark, (cx, cy), 14, (255, 255, 255), 2)
                cv2.circle(image_mark, (cx, cy), 7, (0, 255, 0), -1)
                cv2.circle(mask_image, (cx, cy), 6, (0, 255, 0), -1)
            else :
                cv2.circle(image_mark, (cx, cy), 10, (255, 255, 255), 2)
                cv2.circle(image_mark, (cx, cy), 4, (0, 0, 255), -1)
                cv2.circle(mask_image, (cx, cy), 4, (255, 0, 0), -1)
                counter += 1
            ## Vector of the current object
            vector_object = np.vstack((vector_object, boundingBox))
               
        return image_mark, mask_image, vector_object, counter
        
    
        
        
            
                    
                    
                
                
                
                
                
                
                
                
                
        