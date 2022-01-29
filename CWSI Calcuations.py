# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:06:14 2021

@author: Patron
"""
from matplotlib import image
from matplotlib import pyplot
import math
import numpy as np
from PIL import Image
import cv2 as cv
import numpy.ma as ma

def crop_mask_and_overlay_temps(temps_np, mask_path, at=0, val_sub=0, val_add=0):

    mask_visual = Image.open(mask_path)
    # convert image to numpy array
    mask_np = np.asarray(mask_visual)

    # print(mask_np.shape)

    #shape = mask_np.shape
    # print(shape[0])
    # print(shape[1])
    # print(crop_w)
    # print(crop_h)

    # print("Diff x:",shape[0]- crop_w)
    # print("Diff y:",shape[1]- crop_h)

    # Remove black bounding box from the generated mask
    #mask_np = crop_center(mask_np, shape[1] - crop_h, shape[0] - crop_w)

    mask_np_visual = Image.fromarray(mask_np)

    mask_np_visual.save(mask_path, 'PNG')
    
    # generate binary mask with 1 on non leaf pixels
    not_leaves_mask = np.int64(np.all(mask_np[:, :, :3] == 0, axis=2))

    

    # downscale the binary non sunlit leaf mask to match the thermal image
    downscaled_not_leaves_mask = cv.resize(np.uint8(not_leaves_mask), dsize=(80, 60), interpolation=cv.INTER_CUBIC)

    #Temperature thresholds
    threshold_min = at - val_sub
    threshold_max = at + val_add

    # generate binary mask with 1 where temperature out of threshold
    thermal_thresholding_mask = np.int64(np.logical_or(temps_np< threshold_min, temps_np> threshold_max))

    

    # Logic or between the 2 masks
    final_exclusion_mask_np = thermal_thresholding_mask | downscaled_not_leaves_mask


    # Get the sunlit leaves temperatures only
    temps_np_masked = ma.masked_array(temps_np, mask=final_exclusion_mask_np, fill_value=999)

    print (temps_np)
    # Rebuild 1D array of sunlit leaves using the inverse of the exclusion mask
    sunlit_leaves_only = temps_np_masked[~temps_np_masked.mask]
    sunlit_leaves_mean_temp = sunlit_leaves_only.mean()

    print("Non leaf mask dimensions: ", downscaled_not_leaves_mask.shape)
    print("Out of temperature threshold mask dimensions: ", thermal_thresholding_mask.shape)
    print("Final exclusion mask dimensions: ", final_exclusion_mask_np.shape)
    print("Thermal data np dimensions: ", temps_np_masked.shape)
    print("Leaves only dimensions: ", sunlit_leaves_only.shape)
    print("Mean sunlit leaf temperature: ", sunlit_leaves_mean_temp)
    print("Threshold min:",threshold_min)
    print("Threshold max:",threshold_max)
    print("Atmospheric Temperature: ", at)


    
    # Return the mean sunlit leaf temp and the temperature array with
    # 999 in place of non sulit leaf values
    return sunlit_leaves_mean_temp, temps_np_masked.filled()


def calculateCWSI(Ta,Tc,RH):
    Slope = -1.96
    Intercept = 2.86

    # Ta = 30
    # Tc = 29
    # RH = 0.35

    # Saturation Vapor Pressure at Ta
    VPsat = 0.6108 * math.exp(17.27 * Ta / (Ta + 237.3))

    # Actual Vapor Pressure
    VPair = VPsat * RH/100

    # Vapor Pressure Deficit
    VPD = VPsat - VPair

    # VPsat (Ta + Intercept)
    VPsat_Ta_plus_Intercept = 0.6108 * math.exp(17.27 * (Ta + Intercept) / (Ta + Intercept + 237.3))

    # Vapor Pressure Gradient
    VPG = VPsat - VPsat_Ta_plus_Intercept

    # Temperature difference lower limit
    T_ll = Intercept + Slope * VPD

    # Temperature difference upper limit
    T_ul = Intercept + Slope * VPG

    # Crop Water Stress Index
    CWSI = ((Tc - Ta) - T_ll) / (T_ul - T_ll)

    print("Ta",Ta)
    print("Tc",Tc)
    print("RH",RH)
    print("VPSat: ", VPsat)
    print("VPair: ", VPair)
    print("VPD: ", VPD)
    print("VPsat_Ta_plus_Intercept: ",VPsat_Ta_plus_Intercept)
    print("VPG: ", VPG)
    print("T_ll: ", T_ll)
    print("T_ul: ", T_ul)
    print("CWSI: ", CWSI)

    return CWSI

if __name__=="__main__": 
    """
    at : 2.7 c 
    RH 0.16
    """
    mask_path = 'test.png'
    
    temps = np.loadtxt('temps.csv')
    arr_2d = np.reshape(temps, (60, 80))
    mean_sunlit_temp, leaves_np = crop_mask_and_overlay_temps(arr_2d, mask_path, 36.9, val_sub=7, val_add=7)
    #print(temps)
    print(calculateCWSI(36.9,mean_sunlit_temp,0.16))
        
        
