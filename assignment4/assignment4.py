import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # path = './image2.png'
    path = './image.jpg'
    image = plt.imread(path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # print(gray_image.shape)
    hist_grayscale = cv2.calcHist([gray_image],[0],None,[256],[0,256])

    fig_set = [gray_image, hist_grayscale]
    title_set = ['Grayscale', 'Histogram calcHist']
    types = ['image', 'hist']

    y_axes_values = np.zeros(256,dtype=np.uint32)
    print(y_axes_values.shape)
    # y_axes_values = y_axes_values.flatten()

    i,j = gray_image.shape

    # create histogram manually
    for x in range(i):
        for y in range(j):
            y_axes_values[gray_image[x,y]] = y_axes_values[gray_image[x,y]] + 1
    
    fig_set.append(y_axes_values)
    title_set.append('Histogram manually')
    types.append('hist')

    # neighbourhood processing
    # =============================

    # create the kernel 
    kernel = np.array([[0,0,0],[0,-1,0],[0,0,0]])
    extra_size = int(kernel.shape[0]-1)
    # print(extra_size)

    shell_img = np.zeros((i+extra_size,j+extra_size),dtype=np.uint8)

    # copy the image to the shell
    for x in range(extra_size,i-extra_size):
        for y in range(extra_size,j-extra_size):
            shell_img[x,y] = gray_image[x,y]
    
    # sum of kernel wieights
    sum_kernel = np.sum(kernel)
    
    #apply the mask to the shell image
    for x in range(extra_size,i-extra_size):
        for y in range(extra_size,j-extra_size):
            m=0
            for j in range(-1,1):
                n=0
                temp = 0
                for i in range(-1,1):
                    temp = temp + kernel[m,n]*shell_img[x+i,y+j]
                    n+=1
                m+=1
            shell_img[x,y] =temp/sum_kernel
            
    #==========================================

    # sorry mam, there is an bug in my code and i dont have enough time to fix it at this moment. I would fix it as soon as possible.
    
    #
    
    #
    #
                    
    
    # processed_by_filter = cv2.filter2D(gray_image, -1, kernel)
    # fig_set.append(processed_by_filter)
    # title_set.append('Processed by filter')
    # types.append('image')

    # # print(shell_img)
    # fig_set.append(shell_img)
    # title_set.append('Shell image')
    # types.append('image')

    
    plot_func(types,fig_set,title_set)

def plot_func(types,set,title):
    n = len(set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        plt.subplot(3,3,i+1)
        if types[i] == 'image':
            plt.imshow(set[i],cmap='gray')
        else:
            plt.plot(set[i])
        plt.title(title[i])
    plt.savefig('./output.jpg')
    

main()