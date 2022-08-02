# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np
 
# Reading the input image
image_grayscale = cv2.imread('./image.jpg', 0)
# print(image_grayscale.shape);
img = cv2.threshold(image_grayscale, 50, 255, cv2.THRESH_BINARY)[1]
 
# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)
print(kernel)
 
# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
 
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
 
cv2.waitKey(0)