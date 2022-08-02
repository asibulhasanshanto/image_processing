import matplotlib.pyplot as plt
import cv2

def main():
	path = './image2.jpg'
	image = plt.imread(path)
	
	gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	binary = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)[1]
	
	histogram = cv2.calcHist([gray_image],[0],None,[255],[0,255])	
	printer = {'gray image':gray_image,'binary image':binary}
	
	draw(printer)

def draw(printer):
	n = len(printer)
	plt.figure(figsize=(20,20))
	i=1
	for key,value in printer.items():
		plt.subplot(2,n,i)
		i=i+1
		plt.imshow(value,cmap='gray')
		plt.title(key)
	plt.show()
		

if __name__=='__main__':
	main()