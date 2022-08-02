import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = './image3.jpg'
    image_rgb = plt.imread(img_path)
    w,h = image_rgb.shape[:2]
    print(w,h)

    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask_shell = np.zeros((w,h), dtype = np.uint8)
    
    for i in range(300,900):
        for j in range(200,600):
            mask_shell[i][j] = 255

    outputImage = cv2.bitwise_and(image_grayscale, image_grayscale, mask = mask_shell)
    
    images = [image_rgb, image_grayscale, mask_shell, outputImage]
    titles = ['RGB','Grayscale', 'Mask Shell', 'Output Image']

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
    plt.savefig('./maskOut.jpg')
    # plt.show()


if __name__ == '__main__':
    main()
    