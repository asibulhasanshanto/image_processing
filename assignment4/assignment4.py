import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    # path = './image2.png'
    path = './sunflower.jpg'
    image = plt.imread(path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # print(gray_image.shape)

    # create histogram using openCV
    hist_grayscale = cv2.calcHist([gray_image],[0],None,[256],[0,256])

    fig_set = [gray_image, hist_grayscale]
    title_set = ['Grayscale', 'Histogram calcHist']
    types = ['image', 'hist']

    y_axes_values = np.zeros(256,dtype=np.uint32)

    # create histogram manually
    i,j = gray_image.shape
    for x in range(i):
        for y in range(j):
            y_axes_values[gray_image[x,y]] +=1
    

    
    fig_set.append(y_axes_values)
    title_set.append('Histogram manually')
    types.append('hist')

    # neighbourhood processing manually
    # =============================

    # create the kernel 
    kernel = np.array([[3,4,5],[0,0,0],[-3,-4,-5]])

    # create new row and column
    row_o,col_o = gray_image.shape
    kernel_row,kernel_col = kernel.shape

    new_row = row_o + kernel_row - 1
    new_col = col_o + kernel_col - 1

    # zero padding
    zero_padding = np.zeros((new_row, new_col), dtype=np.uint8)
    r, c = int(kernel_row/2), int(kernel_col/2)
    zero_padding[r:r+row_o, c:c+col_o] = gray_image
    processed_image = np.zeros((row_o, col_o), dtype=np.uint8)

    # convolution
    for x in range(row_o):
        for y in range(col_o):
            mat = zero_padding[x:x+kernel_row, y:y+kernel_col]
            val = np.sum(np.multiply(mat, kernel))
            if val < 0:
                processed_image[x,y] = 0
            elif val > 255:
                processed_image[x,y] = 255
            else:
                processed_image[x,y] = val
    
    fig_set.append(processed_image)
    title_set.append('Neighbourhood processing manually')
    types.append('image')

    # use opencv filter2D
    neighbourhood_processed_cv2 = cv2.filter2D(gray_image, -1, kernel)
    fig_set.append(neighbourhood_processed_cv2)
    title_set.append('Neighbourhood processing cv2')
    types.append('image')
    
    plot_func(types,fig_set,title_set)

def plot_func(types,set,title):
    n = len(set)
    for i in range(n):
        plt.figure(figsize = (15, 10))
        # plt.subplot(3,3,i+1)
        if types[i] == 'image':
            plt.imshow(set[i],cmap='gray')
        else:
            plt.plot(set[i])
        plt.title(title[i])
        plt.savefig('./'+title[i]+'.jpg')
    

main()