import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = './sunflower.jpg'
    image_rgb = plt.imread(img_path)

    w,h = image_rgb.shape[:2]
    print(image_rgb.shape)

    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # binary mask on grayscale image
    #===========================================
    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binary_mask = np.zeros((w,h), dtype = np.uint8)
    
    for i in range(400,675):
        for j in range(700,1000):
            binary_mask[i][j] = 255

    outputImage = cv2.bitwise_and(image_grayscale, image_grayscale, mask = binary_mask)

    images = [image_rgb, image_grayscale, binary_mask, outputImage]
    titles = ['RGB','Grayscale','Binary Mask', 'Binary Masked Image']


    #slice the bits of the grayscale image into 8 image
    #===========================================
    # sliced_1_shell = np.ones((w,h), dtype = np.uint8)
    # sliced_2_shell = np.ones((w,h), dtype = np.uint8)
    # sliced_3_shell = np.ones((w,h), dtype = np.uint8)
    # sliced_4_shell = np.ones((w,h), dtype = np.uint8)
    # sliced_5_shell = np.ones((w,h), dtype = np.uint8)
    # sliced_6_shell = np.ones((w,h), dtype = np.uint8)
    # sliced_7_shell = np.ones((w,h), dtype = np.uint8)
    # sliced_8_shell = np.ones((w,h), dtype = np.uint8)

    # slice_shells = [sliced_1_shell, sliced_2_shell, sliced_3_shell, sliced_4_shell, sliced_5_shell, sliced_6_shell, sliced_7_shell, sliced_8_shell]
    
    # take 8 empty shells
    slice_shells = []
    bit_operators =[1,2,4,8,16,32,64,128]
    for i in range(8):
        slice_shells.append(np.ones((w,h), dtype = np.uint8)* bit_operators[i])
        
    
    # for i in range(8):
    #     slice_shells[i]= slice_shells[i] * bit_operators[i]

    sliced_images = []

    for i in range(8):
        sliced = cv2.bitwise_and(image_grayscale, slice_shells[i], mask =None )
        sliced_images.append(sliced)

    
    images = images + sliced_images
    titles = titles + ['Sliced 1', 'Sliced 2', 'Sliced 3', 'Sliced 4', 'Sliced 5', 'Sliced 6', 'Sliced 7', 'Sliced 8']
    
    # sobel and laplacian apply
    #===========================================
    laplacian = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    
    sobel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    sobel_applied_image = cv2.filter2D(image_grayscale, -1, sobel)
    laplacian_applied_image = cv2.filter2D(image_grayscale, -1, laplacian)

    images = images + [sobel_applied_image, laplacian_applied_image]
    titles = titles + ['Sobel', 'Laplacian']

    plot_img(images, titles)

def plot_img(image_set, title_set):
    n = len(image_set)
    for i in range(n):
        plt.figure(figsize = (15, 10))
        img = image_set[i]
        ch = len(img.shape)

        # plt.subplot( 5, 3, i + 1)
        if (ch == 3):
            plt.imshow(image_set[i])
        else:
            plt.imshow(image_set[i], cmap = 'gray')
        plt.title(title_set[i])
        plt.savefig('./'+title_set[i]+'.jpg')
    # plt.show()


if __name__ == '__main__':
    main()
    