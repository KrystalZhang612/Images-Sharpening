
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from PIL import Image

import time

import math 

import imageio 


#supplemental functions 


#starting by implementing convolution function focusing on kernel of the original image

def ConvolutionFunction(OriginalImage, Kernel):
	
	#the image.shape[0] of the original image represents its height 
	
	ImageHeight = OriginalImage.shape[0]
	
	#the image.shape[1] of the original image represents its width
	
	ImageWidth = OriginalImage.shape[1]
	
	#since kernel here represents a small 2d matrix to blur the original image 
	
	#We represents Kernel.shape[0] as its height, and Kernel.shape[1] as its width 
	
	KernelHeight = Kernel.shape[0]
	
	KernelWidth =  Kernel.shape[1]
	
	#pad numpy arrays within the image
	
	#we consider OriginalImage as an array 
	
	#if the grayscale image gives three element as the number of channels.
	
	
	if(len(OriginalImage.shape) == 3):
		
		PaddedImage = np.pad(OriginalImage, pad_width = ((KernelHeight // 2, KernelHeight // 2), 
		(KernelWidth//2, KernelWidth//2), (0,0)), mode='constant', constant_values=0).astype(np.float32)
		
		
		#if the grayscale image gives two element as the number of channels.
		
		
	elif (len(OriginalImage.shape) == 2):
		
		PaddedImage = np.pad(OriginalImage, pad_width = (( KernelHeight // 2,  KernelHeight // 2),
			(KernelWidth//2, KernelWidth//2)), mode='constant', constant_values=0).astype(np.float32)
		
		
	#floor division result quotient of Kernel height and width divides by 2 
		
	height = KernelHeight // 2
	
	width = KernelWidth // 2
	
	
	#initialize a new array of given shape and type, filled with zeros from padded image 
	
	ConvolvedImage = np.zeros(PaddedImage.shape)
	
	#sum = 0
	
	#iterate the image convolution as 2d array as well 
	
	for i in range(height, PaddedImage.shape[0] - height):
		
		for j in range(width, PaddedImage.shape[1] - width):
			
			
			#2D matrix indexes 
			
			x = PaddedImage[i - height:i-height + KernelHeight, j-width:j-width + KernelWidth]
			
			#use flaten() to return a copy of the array collapsed into one dimension.
			
			x = x.flatten() * Kernel.flatten()
			
			#pass the sum of the array elements into the convolved image matrix
			
			ConvolvedImage[i][j] = x.sum()
			
	#assign endpoints of height and width in the 2D matrix 
			
	HeightEndPoint = -height
	
	WidthEndPoint  = -width 
	
	#when there is no height, return [height, width = width end point] 
	
	if(height == 0):
		
		return ConvolvedImage[height:, width : WidthEndPoint]
	
	#when there is no width, return [height = height end point, width ] 
	
	if(width  == 0):
		
		return ConvolvedImage[height: HeightEndPoint, width:]
	
	#return the convolved image
	
	return ConvolvedImage[height: HeightEndPoint,  width: WidthEndPoint]


#2D Gaussian filter implementation 

def Filter(sigma):
	
	#assign the size of filter 
	
	FilterSize = 2 * int(4 * sigma + 0.5) + 1
	
	#initialize Gaussian filter 
	
	GaussianFilter = np.zeros((FilterSize, FilterSize), np.float32)
	
	#initialize the filter range 
	
	a = FilterSize // 2
	
	b = FilterSize // 2
	
	for m in range(-a, a + 1):
		
		for n in range(-b, b + 1):
			
			#get the area of the circle in the radius of sigma 
			
			m1 = 2 * np.pi*(sigma**2)
			
			#get exponential of (m^2+n^2)/2sigma^2) 
			
			m2 = np.exp(-(m**2 + n**2)/(2* sigma**2))
			
			GaussianFilter[m + a, n + b] = (1/m1)*m2
			
	#return the filter array 
			
	return GaussianFilter

	
#implement the first derivative horizontally of the given sigma value 


def FirstDerivativeX(sigma):
	
	FilterSize = 2 * int(4 * sigma + 0.5) + 1
	
	GaussianFilter = np.zeros((FilterSize, FilterSize), np.float32)
	
	m = FilterSize//2
	
	n = FilterSize//2
	
	for x in range(-m, m+1):
		
		for y in range(-n, n+1):
			
			GaussianFilter[x+m, y+n] = y
			
	return GaussianFilter

#implement the first derivative vertically of the given sigma value 


def FirstDerivativeY(sigma):
	
	FilterSize = 2 * int(4 * sigma + 0.5) + 1
	
	GaussianFilter = np.zeros((FilterSize, FilterSize), np.float32)
	
	m = FilterSize//2
	
	n = FilterSize//2
	
	for x in range(-m, m+1):
		
		for y in range(-n, n+1):
			
			GaussianFilter[x+m, y+n] = x
			
	return GaussianFilter


#Sharpen the image 


def SharpenImage(image, sigma, alpha):
	
	#replace imread(image) with imageio.imread(image):
	
	image = imageio.imread(image)
	
	FilteredImage = np.zeros_like(image, dtype=np.float32)
	
	for k in range(3):
		
		FilteredImage[:, :, k] = ConvolutionFunction(image[:, :, k], FirstDerivativeX(sigma)*FirstDerivativeY(sigma)*Filter(sigma))
		
	return np.clip((image - (alpha * FilteredImage)),0,255).astype(np.uint8)


#similar process as the guassian blur filtering 



#Driver/Testing code:

#Sharpen ”Yosemite.png” with a sigma of 1.0 and alpha of 5.0 and save as ”4.png”.


a = SharpenImage('Yosemite.png', 1.0, 5.0)

plt.imshow(a)

plt.imsave('4.png', a)
