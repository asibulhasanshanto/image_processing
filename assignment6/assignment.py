import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
def main():

    img_path = './image3.jpg'
    print('img_path: ', img_path)
    image_rgb = plt.imread(img_path)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    #average filtering
    kernel = np.ones((3,3),np.uint8)/9
    average_img = cv2.filter2D(img,-1,kernel)
 
    row , col = img.shape
    #salt and pepper noise
    number_of_pixels = int(col)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
      
        img[y_coord][x_coord] = 255
        
    number_of_pixels =int(row)
    for i in range(number_of_pixels):
  
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
    
        img[y_coord][x_coord] = 0

    avg_noise_img = cv2.filter2D(img,-1,kernel)
    gaussian_noise_img = cv2.GaussianBlur(img,(5,5),0)
    median_noise_img = cv2.medianBlur(img,5)
         
    images = [image_rgb,gray,average_img,img,avg_noise_img,gaussian_noise_img,median_noise_img]
    titles = ['RGB','gray','averaging_filter','noise_image','avg_noise_image','gaussian_noise_image','median_noise_image']


    plot_img(images, titles)

def plot_img(image_set, title_set):
    n = len(image_set)
    plt.figure(figsize = (15, 15))
    for i in range(n):
        img = image_set[i]
        ch = len(img.shape)

        plt.subplot( 2, 4, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('./noise.jpg')
    # plt.show()


if __name__ == '__main__':
    main()
    
 