import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = './image3.jpg'
    image_rgb = plt.imread(img_path)

    w,h = image_rgb.shape[:2]

    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_binary = cv2.threshold(image_grayscale, 50, 255, cv2.THRESH_BINARY)[1]

    # create 8 empty images of the same size as the original image
    slicedImage1 = np.zeros((w,h), dtype = np.int8)
    slicedImage2 = np.zeros((w,h), dtype = np.int8)
    slicedImage3 = np.zeros((w,h), dtype = np.int8)
    slicedImage4 = np.zeros((w,h), dtype = np.int8)
    slicedImage5 = np.zeros((w,h), dtype = np.int8)
    slicedImage6 = np.zeros((w,h), dtype = np.int8)
    slicedImage7 = np.zeros((w,h), dtype = np.int8)
    slicedImage8 = np.zeros((w,h), dtype = np.int8)
    slicedImages = [slicedImage1, slicedImage2, slicedImage3, slicedImage4, slicedImage5, slicedImage6, slicedImage7, slicedImage8]

    bit_operators =[1,2,4,8,16,32,64,128]

    for i in range(bit_operators.__len__()):
        for j in range(w):
            for k in range(h):
                if (image_grayscale[j][k] & bit_operators[i]):
                    slicedImages[i][j][k] = 255
                else:
                    slicedImages[i][j][k] = 0
    
    
    
    images = [image_rgb, image_grayscale, image_binary]
    titles = ['RGB','Grayscale', 'Binary']

    images = images + slicedImages
    titles = titles + ['Sliced Image 1', 'Sliced Image 2', 'Sliced Image 3', 'Sliced Image 4', 'Sliced Image 5', 'Sliced Image 6', 'Sliced Image 7', 'Sliced Image 8']

    plot_img(images, titles)

def plot_img(image_set, title_set):
    n = len(image_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = image_set[i]
        ch = len(img.shape)

        plt.subplot( 4, 3, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('./slicedOutput.jpg')
    # plt.show()


if __name__ == '__main__':
    main()
    