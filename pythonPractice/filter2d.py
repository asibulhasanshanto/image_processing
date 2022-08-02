import matplotlib.pyplot as plt
import cv2
def main():
	img_path = './image2.jpg'
	rgb = plt.imread(img_path)
	print(rgb.shape)

	gray_image = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
	print(gray_image.shape)
if __name__=='__main__':
	main()