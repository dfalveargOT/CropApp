#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:30:08 2019
Copyright © 2019 DataRock S.A.S. All rights reserved.
@author: DavidFelipe
"""


try:
    import time
except:
    print(" PLEASE REVIEW THE MODULES THAT NEEDS THE SOFTWARE - AN ERROR WAS OCCURRED")


print(" %% REPORT MODULE %%")
print(" -- Report of counting process  -- Check the current progress --")

class Report:
    
    def Generate(self, name, path, image_time, vector, vector1, vector2, counter):
        """
        Generate the current image report of the counting
        """
        results = open(path + "Results_"+ name +".txt", "w+")
        strE = "DATAROCK S.A.S\n"
        strC = "Copyright © 2019 DataRock S.A.S. All rights reserved.\n"
        strF = time.ctime() + "\n"
        str1 = "Results for " + name + "\n"
        str2 = "Total count : " + str(counter) + "\n"
        str22 = "_ Counter Total : " + str(vector.shape) + "\n"
        str3 = "- Counted L10 : " + str(vector1[0].shape) + "\n"
        str4 = "- Counted L11 : " + str(vector1[1].shape) + "\n"
        str5 = "- Counted L20 : " + str(vector2[0].shape)+ "\n"
        str6 = "- Counted L21 : " + str(vector2[1].shape) + "\n"
        str7 = "Total time of image process : " + str(round(image_time,1)) + "Seconds" + "\n"
        strN = "Software by David Alvear G. \n"
        results.write(strE)
        results.write(strC) 
        results.write(strF)    
        results.write(str1)
        results.write(str2)
        results.write(str22)
        results.write(str3)
        results.write(str4)
        results.write(str5)
        results.write(str6)
        results.write(str7)
        results.write(strN)
        results.close()
        
        