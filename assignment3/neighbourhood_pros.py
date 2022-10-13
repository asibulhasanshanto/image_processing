import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    #'''	Load an RGB image.	'''
    img_path = './image.jpg'
    rgb = cv2.imread(img_path)
    #print(rgb.shape)
        
    #'''	Convert the RGB image into grayscale and binary image.	'''
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    
    #'''	Prepare kernels/filters/masks.	'''
    # kernel1 = np.ones((3, 3), dtype = np.float32) * 2 / 9
    # print('Kernel1: {}'.format(kernel1))

    kernel1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    print('Kernel1: {}'.format(kernel1))	

    kernel2 = np.array([[1, 2, 3], [0, 0, 0], [-1,-2,-3]])
    print('Kernel2: {}'.format(kernel2))

    kernel3 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    print('Kernel3: {}'.format(kernel3))
    
    kernel4 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    print('Kernel4: {}'.format(kernel4))
    
    kernel5 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    print('Kernel5: {}'.format(kernel5))

    kernel6 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    print('Kernel6: {}'.format(kernel6))
    
    #'''	Neighborhood processing. '''
    processed_img1 = cv2.filter2D(grayscale, -1, kernel1)
    processed_img2 = cv2.filter2D(grayscale, -1, kernel2)
    processed_img3 = cv2.filter2D(grayscale, -1, kernel3)
    processed_img4 = cv2.filter2D(grayscale, -1, kernel4)
    processed_img5 = cv2.filter2D(grayscale, -1, kernel5)
    processed_img6 = cv2.filter2D(grayscale, -1, kernel6)
        
    #'''	Plot images. '''
    img_set = [rgb, grayscale, processed_img1, processed_img2,processed_img3,processed_img4,processed_img5,processed_img6]
    title_set = ['RGB', 'Grayscale', 'Kernel1', 'Kernel2','Kernel3','Kernel4','Kernel5','Kernel6']
    plot_img(img_set, title_set)

def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
    
        plt.subplot(3, 3, i + 1)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.show()		
    
if __name__ == '__main__':
    main()