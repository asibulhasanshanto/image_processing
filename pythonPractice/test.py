import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
	image_path = './image2.jpg'
	image = plt.imread(image_path)
	print(image.shape)
		
	gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	gray_image_histogram = cv2.calcHist([gray_image],[0],None,[255],[0,255])
	print(gray_image[10][10])

	binary_image = cv2.threshold(gray_image,50,255,cv2.THRESH_BINARY)[1]
	binary_histogram = cv2.calcHist([binary_image],[0],None,[255],[0,255])

	red_image = image[:, :, 0]
	green_image = image[:, :, 1]
	blue_image = image[:, :, 2]

	red_image_histogram = cv2.calcHist([red_image],[0],None,[255],[0,255])
	green_image_histogram = cv2.calcHist([green_image],[0],None,[255],[0,255])
	blue_image_histogram = cv2.calcHist([blue_image],[0],None,[255],[0,255])

	laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	sharpen = cv2.filter2D(gray_image,-1,laplacian)
	sharpen_histogram = cv2.calcHist([sharpen],[0],None,[255],[0,255])

	

	image_dictionary = {'Gray':[gray_image,gray_image_histogram],'Binary':[binary_image ,binary_histogram ],'RED':[red_image ,red_image_histogram],'Sharpen':[sharpen,sharpen_histogram]}

	print_images(image_dictionary)

def print_images(dictionary):
	n = len(dictionary)
	plt.figure(figsize=(20,20))
	
	i=1
	for key,value in dictionary.items():
		plt.subplot(n,2,i)
		plt.imshow(value[0],cmap='gray')
		plt.title(key)

		plt.subplot(n,2,i+1)
		plt.xlim([-10,300])
		plt.plot(value[1])
		plt.title(key)
		i+=2
	plt.show()
		
	
if __name__=='__main__':
	main()