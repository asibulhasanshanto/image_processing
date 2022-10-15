import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    rgbImg = plt.imread('sunflower.jpg')
    print(rgbImg.shape)
    
    # take the original image
    gray =  cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)

    # apply fast fourier transform
    image_in_frequency_domain = np.fft.fft2(gray)
    # shift the zero-frequency component to the center of the spectrum
    centered_f_img = np.fft.fftshift(image_in_frequency_domain)
    magnitude_spectrum = 100 * np.log(np.abs(image_in_frequency_domain))
    centered_magnitude_spectrum = 100 * np.log(np.abs(centered_f_img))
    
    # prepare the filter
    r , c = gray.shape
    shell_image = np.ones((r,c),dtype=np.uint8)
    
    # filter = cv2.circle(shell_image,(int(c/2),int(r/2)),30,(0,0,0),-1) #high pass filter
    # filter = cv2.circle(shell_image,(int(c/2),int(r/2)),30,(255,255,255),-1)#low pass filter
    filter = cv2.line(shell_image,(0,int(r/2)),(c,int(r/2)),(0,0,0),9) # high pass line
    # filter = cv2.line(shell_image,(int(c/2),0),(int(c/2),c),(255,255,255),9) #low pass line

    # apply the filter
    filter_applied_f_img = centered_f_img * filter

    # inverse fourier transform
    filtered_img = np.abs(np.fft.ifft2(filter_applied_f_img))   


    # filter2 = cv2.circle(white_img.copy(),(m,n),25,(0,0,0),-1)
    # filter3 = cv2.line(white_img.copy(),(m,0),(m,c),(255,255,255),9)
    # filter4 = cv2.line(white_img.copy(),(0,n),(r,n),(0,0,0),9)

    

    
    img_set = [magnitude_spectrum, centered_magnitude_spectrum, filter, filtered_img]
    title_set = ['FFT2 of image', 'Centered FFT2 of image', 'High pass line filter', 'Filtered Img']
    
    plot_img(img_set,title_set)
    
def plot_img(img_set, title_set):		
    plt.figure(figsize = (10,8))
    n = len(img_set)
    for i in range(n):
        plt.subplot(2, 2, i + 1)
        plt.title(title_set[i])
        img = img_set[i]
        ch = len(img.shape)
        if (ch == 2):
            plt.imshow(img, cmap = 'gray')
        else:
            plt.imshow(img)			
    plt.savefig('high_pass_line_filter.png')
    # plt.show()





if __name__ == '__main__':
    main()