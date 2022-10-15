import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    images = []
    titles = []

    img_path = './morph.jpg'
    image_rgb = plt.imread(img_path)

    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binary_image = cv2.threshold(image_grayscale, 127, 255, cv2.THRESH_BINARY)[1]
    structuring_element = np.array([[0,0,1,0],[0,1,1,0],[1,1,1,1],[0,0,1,0],[0,0,1,0]],np.uint8)
    morph_rect = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    morph_ellipse = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
    morph_cross = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(5, 5))
    structuring_element = morph_cross

    img_erosion = cv2.erode(binary_image, structuring_element, iterations=2)
    img_dilation = cv2.dilate(binary_image, structuring_element, iterations=2)
    img_opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, structuring_element)
    img_closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, structuring_element)


    # images.append(image_grayscale)
    # titles.append('Grayscale')
    images.append(img_erosion)
    titles.append('Erosion with cross')
    images.append(img_dilation)
    titles.append('Dilation with cross')
    images.append(img_opening)
    titles.append('Opening with cross')
    images.append(img_closing)
    titles.append('Closing with cross')

    plot_img(images, titles)



def plot_img(image_set, title_set):
    n = len(image_set)
    plt.figure(figsize = (10, 8))
    for i in range(n):
        img = image_set[i]
        ch = len(img.shape)

        plt.subplot( 2, 2, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
        plt.savefig('./Output_cross.jpg')
    # plt.show()

if __name__ == '__main__':
    main()