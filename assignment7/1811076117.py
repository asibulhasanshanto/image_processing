import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
	image_path = './image1.jpg'
	rgb_image = plt.imread(image_path)
	
	grayScaleImage = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)
	grayScaleHistogram = cv2.calcHist([grayScaleImage],[0],None,[255],[0,255])
	
	leftShiftedImage = shiftHistogram(grayScaleImage ,-100,[0,255])
	rightShiftedImage = shiftHistogram(grayScaleImage ,50,[0,255])
	shiftInRange = shiftHistogram(grayScaleImage ,30,[100,200])
	
	drawing_list = {"Gray Scale ":[grayScaleImage,grayScaleHistogram],"Right Shifted":[rightShiftedImage[0],rightShiftedImage[1] ],"Left Shifted":[leftShiftedImage[0],leftShiftedImage[1]],"Shift in Range":[shiftInRange[0],shiftInRange[1]]}

	draw_figs(drawing_list)
	
def draw_figs(drawing_list):
	n = len(drawing_list)
	plt.figure(figsize=(30,30))
	i=1
	
	for key,value in drawing_list.items():
		plt.subplot(n,2,i)
		plt.imshow(value[0],cmap='gray')
		plt.title(key + ' image')

		plt.subplot(n,2,i+1)
		plt.plot(value[1])
		plt.title(key + ' image histogram')
		
		i+=2
	
	#plt.show()
	plt.savefig('./output2.jpg')

def shiftHistogram(image,distance,limit):
	histogram = np.zeros(256,dtype=np.uint16)
	new_image = np.copy(image)
	r,c = image.shape
	print(distance)
	for i in range(r):
		for j in range(c):
			if(image[i][j]>=limit[0] and image[i][j]<=limit[1]):
				temp = image[i][j] + distance
				if temp>limit[1]:
					#new_image[i][j]=limit[0]+ temp-limit[1]-1 #use this line for rounding values
					new_image[i][j]=limit[1]
				elif temp<limit[0]:
					#new_image[i][j]=limit[1]+ temp-limit[0]+1 #use this line for rounding values
					new_image[i][j]=limit[0]
				else:
					new_image[i][j] = temp
				
			if(histogram[new_image[i][j]]>=30000):
				histogram[new_image[i][j]]=30000
			else:
				histogram[new_image[i][j]]+=1
	return [new_image,histogram]

if __name__=='__main__':
	main()