import matplotlib.pyplot as plt
import cv2


def main():
    img_path = './test.jpg'
    print(img_path)

    img =plt.imread(img_path)
    # print("normal image infos")
    # print(img.shape)
    # print(img)

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("grayscale image infos")
    print(grayscale_img.shape)

    binary_img = cv2.cvtColor(grayscale_img, cv2.THRESH_BINARY)
    print("binary image infos")
    print(binary_img.shape)

    plt.figure(figsize = (20,20))
    plt.subplot(2,3,1)
    plt.title('RGB')
    plt.imshow(img)
    
    plt.subplot(2,3,2)
    plt.title('RED')
    plt.imshow(img[:, :, 0])

    plt.subplot(2,3,3)
    plt.title('GREEN')
    plt.imshow(img[:, :, 1])

    plt.subplot(2,3,4)
    plt.title('BLUE')
    plt.imshow(img[:, :, 2])

    plt.subplot(2,3,5)
    plt.title('GRAYSCALE')
    plt.imshow(grayscale_img, cmap = 'gray')

    plt.subplot(2,3,5)
    plt.title('BINARY')
    plt.imshow(binary_img, cmap = 'gray')

    plt.show()

if __name__ == '__main__':
    main()