#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:19:37 2019
Copyright © 2019 DataRock S.A.S. All rights reserved.
@author: DavidFelipe
"""

import Preprocessing
import MatchCore
import MarkProcess
import Classification
import Report
import os
import cv2
import time
import numpy as np
import shutil

init_time = time.time()
print('\n')
print("DATAROCK S.A.S")
print("CROPS COUNTING SOFTWARE")
print("Copyright © 2019 DataRock S.A.S. All rights reserved.")
print("Developed by David Felipe Alvear Goyes")
time.sleep(2)
path_images = "./ToProcess/"
elements_toProcess = os.listdir(path_images)
path_processed = "./Results/"
length_images = len(elements_toProcess)
print("Objects found to be processed : ")
print(elements_toProcess)
print('\n')
time.sleep(3)
counter_images = 0
shutil.rmtree("Results")
os.mkdir("Results")

for element in elements_toProcess:
    
    jpg_flag = element.find(".jpg")
    png_flag = element.find(".png")
    jpeg_flag = element.find(".jpeg")
    counter_images += 1
    image_time = time.time()
    if(jpg_flag != -1 or png_flag != -1 or jpeg_flag != -1):
        
        print('\n')
        image_time = time.time()
        print("STARTING IMAGE PROCESSING")
        print("Current image : " + element)
        print("Object # "+ str(counter_images) + " of " + str(length_images))
        time.sleep(3)
        image = cv2.imread(path_images + element)
        """
        Create Directories to save information
        """
        os.mkdir("Results/"+element)
        image_path = element + "/"
        """
        Module one - preprocessing
        """
        ### Read the image for entry in the first module
        print("Preprocessing layers of image process model")
        preprocess = Preprocessing.preprocessing(95, 98, image)
        layer1 = preprocess.processingLayer1()
        layer2 = preprocess.processingLayer2()
        layer3, layer3_raw, rangeFourier, meanFourier = preprocess.processingLayer3()
        
        ### Processing the information of the  below layers
        print("Object matching layers of image process model")
        matchObject = MatchCore.MatchCore(image,layer1, layer2, layer3, meanFourier, rangeFourier)
        Mlayer10, Mlayer11, Mlayer12, maskLayer1, Vlayer1, blobLayer1 = matchObject.CoreFinding(layer1)
        Mlayer20, Mlayer21, Mlayer22, maskLayer2, Vlayer2, blobLayer2 = matchObject.CoreFinding(layer2)
        Mlayer3, maskLayer3, Vlayer3, blobLayer3 = matchObject.CoreFinding(layer3)
        
        ### Determine the objects to pass to the Machine learning phase
        print("Image match objects of processing model ")
        markProcess = MarkProcess.MarkProcess(image, 4, 0)
        vectorL1, vectorL2, vectorL3 = markProcess.objectMatch(Vlayer1, Vlayer2, Vlayer3)
        imageL10 = markProcess.imageMatch(vectorL1[0])
        imageL11 = markProcess.imageMatch(vectorL1[1])
        imageL12 = markProcess.imageMatch(vectorL1[2])
        imageL20 = markProcess.imageMatch(vectorL2[0])
        imageL21 = markProcess.imageMatch(vectorL2[1])
        imageL22 = markProcess.imageMatch(vectorL2[2])
        imageL3 = markProcess.imageMatch(vectorL3)
        meanArea1 = markProcess.meanGeometry(vectorL1)
        meanArea2 = markProcess.meanGeometry(vectorL2)
        meanArea3 = markProcess.meanGeometry(vectorL3)
        
        """
        Las sisguientes Capas no se activaran
        Layer 1 - subcapa 2
        Layer 2 - subcapa 2
        """
        print(" ")
        print("Converge layer objectmatching in the processing model")
        image_group, image_mask, Layer_Group = markProcess.Vagroup(vectorL1[0], vectorL1[1], vectorL2[0], vectorL2[1])
        image_mark, object_mask, vector_object, counter = markProcess.countIntegration(image, image_mask)
        print(" ")
        print("Writting the information on the disk")
        cv2.imwrite(path_processed + image_path + "Marked_"+ element, image_mark.astype('uint8'))
        cv2.imwrite(path_processed + image_path + "Mask_"+ element, image_mask.astype('uint8'))
        cv2.imwrite(path_processed + image_path + "L10_"+ element, imageL10.astype('uint8'))
        cv2.imwrite(path_processed + image_path + "L11_"+ element, imageL11.astype('uint8'))
        cv2.imwrite(path_processed + image_path + "L20_"+ element, imageL20.astype('uint8'))
        cv2.imwrite(path_processed + image_path + "L21_"+ element, imageL21.astype('uint8'))
        
        ## Reporte
        print("Agroup information for report the process")
        name = element[0:element.find(".")]
        time_end = time.time() - image_time
        report_image = Report.Report()
        report_image.Generate(name, path_processed, time_end, vector_object, vectorL1, vectorL2, counter)
    
        print("Ending process for image " + element)
        print("Image time processing : " + str(time.time() - image_time) + "Seconds") 
    #    break
    else:
        print("WARNING ")
        print(element + " Not a image file allowed")
    
Total_time = time.time() - init_time
print("TOTAL TIME FOR THE ELEMENTS " + str(Total_time) + "Seconds") 
finish = input("Push any key to finish ")
    
