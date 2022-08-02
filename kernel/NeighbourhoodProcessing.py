import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # Read in the image
    img_path = './image.jpg'
    image_rgb = plt.imread(img_path)


    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    #neighbourhood processing
    kernel1 = np.ones((3,3),dtype=np.int8)
    processed_img1 = cv2.filter2D(image_grayscale,-1,kernel1)

    kernel2 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=np.int8)
    processed_img2 = cv2.filter2D(image_grayscale,-1,kernel2)


    # Plot the images
    images = [image_grayscale,processed_img1,processed_img2]
    titles = ['Grayscale','Kernel1','Kernel2']

    plot_img(images,titles)

def plot_img(img_set,title_set):
    n=len(img_set)

    plt.figure(figsize=(20, 20))
    for i in range(n):
        plt.subplot(2, 3, i + 1)
        plt.title(title_set[i])
        plt.imshow(img_set[i], cmap='gray')
    plt.show()

   

if __name__=='__main__':
    main()