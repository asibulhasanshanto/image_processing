from cv2 import equalizeHist
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


def main():
    rgb = cv2.imread('sunflower.jpg')
    print(rgb.shape)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(grayscale)

    # create histogram
    hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    equalizedhist = cv2.calcHist([result], [0], None, [256], [0, 256])
    
    # plot histogram
    plt.figure(figsize=(15, 10))
    plt.plot(hist, color='r')
    plt.title('Normal Image Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Number of pixels')
    plt.savefig('histogram.png')
    plt.figure(figsize=(15, 10))
    plt.plot(equalizedhist, color='b')
    plt.title('Equalized Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Number of pixels')
    plt.savefig('histogram_equalized.png')

    img_set = [grayscale, result]
    title_set = ['Original Image', 'Equalize Image']

    plot_img(img_set, title_set)


def plot_img(img_set, title_set):
    n = len(img_set)

    plot_number = 1
    for i in range(n):
        # plt.subplot(2, 2, plot_number)
        plt.figure(figsize=(15, 10))
        plt.title(title_set[i])

        ch = len(img_set[i].shape)
        if ch == 3:
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap='gray')

        plt.savefig('./'+title_set[i] +'.png')
        # plt.subplot(2, 2, plot_number + 1)
        plt.hist(img_set[i].ravel(), 256, [0, 256])
        plt.title(title_set[i] + ' Histogram')
        plt.xlabel('Intensity')
        plt.ylabel('Number of pixels')

        plot_number += 2



def rgb_to_grayscale(rgb):
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    graysclae = red * 0.2989 + green * 0.5870 + blue * 0.1140
    graysclae = graysclae.astype(int)

    return graysclae


def hist_equal(img):
    row, col = img.shape
    num_of_pixels = []
    equalization = []

    img_1d = img.reshape(-1)
    num_of_pixels = np.zeros((256,), dtype=int)

    for i in range(len(img_1d)):
        num_of_pixels[img_1d[i]] += 1

    for i in range(256):
        pdf = num_of_pixels[i]/np.sum(num_of_pixels)

        if i == 0:
            cdf = pdf
        else:
            cdf = cdf + pdf

        equalization.append(math.ceil(cdf * 255))

    result = np.zeros((row, col), dtype=np.uint8)

    for i in range(row):
        for j in range(col):
            result[i, j] = equalization[img[i, j]]

    return result


if __name__ == '__main__':
    main()