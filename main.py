#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
from Inpainter import Inpainter

if __name__ == "__main__":
    if not len(sys.argv) == 3 and not len(sys.argv) == 4:
        print(
            'Usage: python main.py pathOfInputImage pathOfMaskImage [,halfPatchWidth].')
        exit(-1)

    if len(sys.argv) == 3:
        halfPatchWidth = 4
    elif len(sys.argv) == 4:
        try:
            halfPatchWidth = int(sys.argv[3])
        except ValueError:
            print('Unexpected error:', sys.exc_info()[0])
            exit(-1)
    #
    # halfPatchWidth = 4
    # image File Name
    # originalImage = cv2.imread("Input_resize.jpg", 1)
    # originalImage = cv2.resize(originalImage,(1000,1000))
    # cv2.imshow('w', originalImage)
    # CV_LOAD_IMAGE_COLOR: loads the image in the RGB format TODO: check RGB sequence
    imageName = sys.argv[1]
    originalImage = cv2.imread(imageName, 1)
    if originalImage is None:
        print('Error: Unable to open Input image.')
        exit(-1)
    # ratatouille_resize.jpg msk_out.jpg
    # mask File Name
    # inpaintMask = cv2.imread("msk_out.jpg", 0)
    # inpaintMask = cv2.resize(inpaintMask, (1000, 1000))
    maskName = sys.argv[2]
    inpaintMask = cv2.imread(maskName, 0)
    if inpaintMask is None:
        print('Error: Unable to open Mask image.')
        exit(-1)

    i = Inpainter(originalImage, inpaintMask, halfPatchWidth)
    if i.checkValidInputs() == i.CHECK_VALID:
        i.inpaint()
        cv2.imwrite("result.png", i.result)
        # cv2.namedWindow("result")
        # cv2.imshow("result", i.result)
        # cv2.waitKey()
    else:
        print('Error: invalid parameters.')
