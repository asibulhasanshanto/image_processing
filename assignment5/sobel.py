import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
img0 = plt.imread('sunflower.jpg')
print(img0)

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)

# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)



# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_32F)
sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)  # x
sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()