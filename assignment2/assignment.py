
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

def main():
    img_path = './whiteFlower.jpg'
    rgb_image =  plt.imread(img_path)
    print(rgb_image.shape)

    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    print(grayscale_image.shape)

    # take the shape of grayscale image and create empty images
    i,j = grayscale_image.shape
    processed_image1 = np.zeros((i,j), dtype = np.uint8)
    processed_image2 = np.zeros((i,j), dtype = np.uint8)
    processed_image3 = np.zeros((i,j), dtype = np.uint8)
    processed_image4 = np.zeros((i,j), dtype = np.uint8)

    # take threshold values and other parameters
    T1=100
    T2=200
    c =2
    p=5
    epsilon =  0.0000001

    # process the image
    for x in range(i):
        for y in range(j):
            if grayscale_image[x,y]>=T1 and grayscale_image[x,y]<=T2:
                processed_image1[x,y] = 100
            else:
                processed_image1[x,y] = 10
            
            if grayscale_image[x,y]>=T1 and grayscale_image[x,y]<=T2:
                processed_image2[x,y] = 100
            else:
                processed_image2[x,y] = grayscale_image[x,y]
            
            processed_image3[x,y] =  c * np.log(1+grayscale_image[x,y])
            processed_image4[x,y] = c * pow((grayscale_image[x,y] + epsilon),p)
    
    image_set=[rgb_image, grayscale_image, processed_image1, processed_image2, processed_image3, processed_image4]
    title_set=['RGB', 'Grayscale', 'Processed 1', 'Processed 2', 'Processed 3', 'Processed 4']
    plot_img(image_set, title_set)

def plot_img(image_set, title_set):
    n = len(image_set)
    for i in range(n):
        plt.figure(figsize = (10, 15))
        img = image_set[i]
        ch = len(img.shape)

        # plt.subplot( 2, 3, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
        plt.savefig('./'+title_set[i]+'.jpg')

if __name__ == '__main__':
    main()