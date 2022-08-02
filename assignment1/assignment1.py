import matplotlib.pyplot as plt
import cv2

def main():
    # Read in the image
    img_path = './image.jpg'
    image_rgb = plt.imread(img_path)

    # Convert to red green and blue channels and also in grayscal and in binary
    image_red = image_rgb[:, :, 0]
    image_green = image_rgb[:, :, 1]
    image_blue = image_rgb[:, :, 2]

    image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_binary = cv2.threshold(image_grayscale, 50, 255, cv2.THRESH_BINARY)[1]

    # Plot the images
    images = [image_rgb, image_red, image_green, image_blue, image_grayscale, image_binary]
    titles = ['RGB', 'Red', 'Green', 'Blue', 'Grayscale', 'Binary']

    plt.figure(figsize=(20, 20))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.title(titles[i])
        ch = len(images[i].shape)
        if ch == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i],cmap='gray')
    plt.savefig('./image_figure')

    #calculate the histogram of the image
    hist_grayscale = cv2.calcHist([image_grayscale],[0],None,[256],[0,256])
    hist_red = cv2.calcHist([image_rgb],[2],None,[256],[0,256])
    hist_green = cv2.calcHist([image_rgb],[1],None,[256],[0,256])
    hist_blue = cv2.calcHist([image_rgb],[0],None,[256],[0,256])
    hist_binary = cv2.calcHist([image_binary],[0],None,[256],[0,256])

    histogram_images = [hist_grayscale, hist_red, hist_green, hist_blue, hist_binary]
    histogram_titles = ['Grayscale', 'Red', 'Green', 'Blue', 'Binary']

    plt.figure(figsize=(30, 20))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.title(histogram_titles[i])
        plt.plot(histogram_images[i])
        if histogram_titles[i] == 'Binary':
            plt.xlim([-10, 260])
        else:
            plt.xlim([0, 256])
    plt.savefig('./hist_figure')

if __name__=='__main__':
    main()