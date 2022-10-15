import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def main():

    img_path = './watch.jpg'
    print('img_path: ', img_path)
    image_rgb = plt.imread(img_path)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # noisy_image=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # average_img = cv2.filter2D(img,-1,kernel)
 
    #configs
    row , col = gray.shape
    number_of_pixels = int((row * col)/30)
    noisy_image = gray.copy()

    #adding salt and pepper noise
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        noisy_image[y_coord][x_coord] = 255
        
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        noisy_image[y_coord][x_coord] = 0

    #average filtering
    kernel = np.ones((3,3),np.uint8)/9

    # removing noise 
    avg_noise_img = cv2.filter2D(noisy_image,-1,kernel)
    gaussian_noise_img = cv2.GaussianBlur(noisy_image,(5,5),0)
    median_noise_img = cv2.medianBlur(noisy_image,5)
         
    images = [image_rgb,gray,noisy_image,avg_noise_img,gaussian_noise_img,median_noise_img]
    titles = ['RGB Image','Gray Image','Noisy image','Average filter applied','Gaussien filter applied','Median filter applied']


    plot_img(images, titles)

def plot_img(image_set, title_set):
    n = len(image_set)
    for i in range(n):
        plt.figure(figsize = (10, 10))
        img = image_set[i]
        ch = len(img.shape)

        # plt.subplot( 3, 3, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
        plt.savefig('./'+title_set[i]+'.jpg')
    # plt.show()


if __name__ == '__main__':
    main()
    
 