import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = './image3.jpg'
    image_rgb = plt.imread(img_path)

    w,h = image_rgb.shape[:2]
    print(image_rgb.shape)

    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # binary mask on grayscale image
    #===========================================
    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binary_mask = np.zeros((w,h), dtype = np.uint8)
    
    for i in range(300,900):
        for j in range(200,600):
            binary_mask[i][j] = 255

    outputImage = cv2.bitwise_and(image_grayscale, image_grayscale, mask = binary_mask)

    images = [image_rgb, image_grayscale, binary_mask, outputImage]
    titles = ['RGB','Grayscale','Binary Mask', 'Binary Masked Image']


    #slice the bits of the grayscale image into 8 image
    #===========================================
    sliced_1_shell = np.ones((w,h), dtype = np.uint8)
    sliced_2_shell = np.ones((w,h), dtype = np.uint8)
    sliced_3_shell = np.ones((w,h), dtype = np.uint8)
    sliced_4_shell = np.ones((w,h), dtype = np.uint8)
    sliced_5_shell = np.ones((w,h), dtype = np.uint8)
    sliced_6_shell = np.ones((w,h), dtype = np.uint8)
    sliced_7_shell = np.ones((w,h), dtype = np.uint8)
    sliced_8_shell = np.ones((w,h), dtype = np.uint8)

    slice_shells = [sliced_1_shell, sliced_2_shell, sliced_3_shell, sliced_4_shell, sliced_5_shell, sliced_6_shell, sliced_7_shell, sliced_8_shell]
    bit_operators =[1,2,4,8,16,32,64,128]
    
    for i in range(8):
        slice_shells[i]= slice_shells[i] * bit_operators[i]

    sliced_images = []

    for i in range(8):
        sliced = cv2.bitwise_and(image_grayscale, slice_shells[i], mask =None )
        sliced_images.append(sliced)

    
    images = images + sliced_images
    titles = titles + ['Sliced Image 1', 'Sliced Image 2', 'Sliced Image 3', 'Sliced Image 4', 'Sliced Image 5', 'Sliced Image 6', 'Sliced Image 7', 'Sliced Image 8']
    

    plot_img(images, titles)

def plot_img(image_set, title_set):
    n = len(image_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = image_set[i]
        ch = len(img.shape)

        plt.subplot( 5, 3, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('./Output.jpg')
    # plt.show()


if __name__ == '__main__':
    main()
    