
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

def main():
    img_path = './image.jpg'
    rgb_image =  plt.imread(img_path)
    print(rgb_image.shape)

    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    print(grayscale_image.shape)

    i,j = grayscale_image.shape
    processed_image1 = np.zeros((i,j), dtype = np.uint8)
    processed_image2 = np.zeros((i,j), dtype = np.uint8)
    processed_image3 = np.zeros((i,j), dtype = np.uint8)

    T1=100
    T2=200
    c1 = 255/math.log(256)
    c2 = math.log(255)/255
    epsilon =  0.0000001
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
            
            processed_image3[x,y] = c1 * math.log(grayscale_image[x,y]+1)
    
    image_set=[rgb_image, grayscale_image, processed_image1, processed_image2, processed_image3]
    title_set=['RGB', 'Grayscale', 'Processed1', 'Processed2', 'Processed3']
    plot_img(image_set, title_set)

def plot_img(image_set, title_set):
    n = len(image_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = image_set[i]
        ch = len(img.shape)

        plt.subplot( 2, 3, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('./image_processed.jpg')

if __name__ == '__main__':
    main()