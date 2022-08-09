import matplotlib.pyplot as plt
import cv2

def main():
	path = './image1.jpg'
	image = plt.imread(path)
	gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

	histogram = cv2.calcHist([gray_image],[0],None,[255],[0,255])
	equalized_image = cv2.equalizeHist(gray_image)
	histogram_equalized = cv2.calcHist([equalized_image],[0],None,[255],[0,255])
	
	printer = {'Gray histogram':['tut',histogram_equalized]}
	
	draw(printer)

def draw(printer):
	n = len(printer)
	plt.figure(figsize=(20,20))
	i=1
	for key,value in printer.items():
		plt.subplot(n,2,i)
		i=i+1
		if value[0]=='image':
			plt.imshow(value[1],cmap='gray')
		else:
			plt.plot(value[1])
		plt.title(key)
	plt.show()
		

if __name__=='__main__':
	main()