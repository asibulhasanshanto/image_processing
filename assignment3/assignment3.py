import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    image_path = './image3.jpg'
    rgb_image = plt.imread(image_path)

    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    kernel1 = np.ones((3, 3), dtype = np.float32)*25/100
    # print(kernel1)

    kernelBlur = np.array([[0,0.4,0],[0.4,-0.9,0.4],[0,0.4,0]])
    kernel3 = np.ones((3, 3), dtype = np.float32)*-0.5
    kernel3[1,1]=4

    kernel4 = np.array([[3,4,5],[0,0,0],[-3,-4,-5]])
    kernel5 = np.array([[-3,-2,-1],[-1,1,1],[1,2,3]])
    kernel6 = np.array([[-1,0,-1],[0,8,0],[-1,0,-1]])

    kernels = [kernel1,kernelBlur,kernel3,kernel4,kernel5,kernel6]
    kernelNames=['Kernel1','kernel blur','kernel3','kernel4','kernel5','kernel6']
    processed_images = [rgb_image,grayscale_image]
    processed_images_titlels = ['RGB Image','Grayscale Image']

    c = 0
    for kernel in kernels:
        processed_image = cv2.filter2D(grayscale_image, -1, kernel)
        processed_images.append(processed_image)
        processed_images_titlels.append(kernelNames[c])
        c+=1
        
    plot_images(processed_images, processed_images_titlels)

def plot_images(images,titles):
    n = len(images)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = images[i]
        ch = len(img.shape)

        plt.subplot(3, 3, i + 1)
        if (ch == 3):
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap = 'gray')
        plt.title(titles[i])
    plt.savefig('./output')
    # plt.show()
if __name__ == '__main__':
    main()