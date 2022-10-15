import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
	image_path = './sunflower2.jpg'
	rgb_image = plt.imread(image_path)
	
	grayScaleImage = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)
	grayScaleHistogram = cv2.calcHist([grayScaleImage],[0],None,[255],[0,255])
	
	# get the highest value of histogram
	highestValue = np.amax(grayScaleHistogram)

	leftShiftedImage = shiftHistogram(grayScaleImage ,-100,[0,255],highestValue)
	rightShiftedImage = shiftHistogram(grayScaleImage ,50,[0,255],highestValue)
	shiftInRange = shiftHistogram(grayScaleImage ,30,[100,200],highestValue)
	
	drawing_list = {"Gray Scale ":[grayScaleImage,grayScaleHistogram],"Right Shifted":[rightShiftedImage[0],rightShiftedImage[1] ],"Left Shifted":[leftShiftedImage[0],leftShiftedImage[1]],"Shift in Range":[shiftInRange[0],shiftInRange[1]]}

	draw_figs(drawing_list)
	
def draw_figs(drawing_list):
	n = len(drawing_list)
	i=1
	
	for key,value in drawing_list.items():
		plt.figure(figsize=(15,10))
		# plt.subplot(n,2,i)
		plt.imshow(value[0],cmap='gray')
		plt.title(key + ' image')
		plt.savefig('./'+key + ' image'+'.jpg')

		plt.figure(figsize=(15,10))
		# plt.subplot(n,2,i+1)
		plt.plot(value[1])
		plt.title(key + ' image histogram')
		plt.savefig('./'+key + ' image histogram'+'.jpg')
		
		i+=2
	
	#plt.show()

def shiftHistogram(image,distance,limit,highValue):
    
	histogram = np.zeros(256,dtype=np.uint16)
	new_image = np.copy(image)
	r,c = image.shape
	print(image.shape)
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
				
			if(histogram[new_image[i][j]]>=highValue):
				histogram[new_image[i][j]]=highValue
			else:
				histogram[new_image[i][j]]+=1
	return [new_image,histogram]

if __name__=='__main__':
	main()