import numpy as np
import cv2
import argparse
import math

def main(args):
	image = cv2.imread(args.pathToImage)
	# use hough line transform to determine the location of lines in the input image
	lines = parseImage(args.pathToImage)
	# convert the lines from parametric to cartesian coordinates and determine the bounds of each line
	finalLines = determineBounds(image, lines)
	# write these lines to a cp file
	cpFileName = args.pathToImage.split('.')[0] + '.cp'
	writeCP(cpFileName, finalLines)
	
def parseImage(pathToImage):
	# read image
	#image = cv2.resize(cv2.imread(pathToImage), (400,400))
	image = cv2.imread(pathToImage)
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY)
	grayscale = thresh
	#showImage(image, 'image')
	showImage(grayscale, 'grayscale')
	# perform line detection
	edges = cv2.Canny(grayscale, 50, 150, apertureSize=3)
	#print(edges)
	lines = cv2.HoughLines(edges,1,np.pi/180,400)
	blank = np.zeros(grayscale.shape)
	print(lines)
	degreesToRho = {}
	for line in lines:
		for rho,theta in line:

			#print('rho', rho)
			#print('theta', theta)
			degrees = round((360 * theta) / (2 * math.pi), 3)
			if (degreesToRho.get(degrees) is None):
				degreesToRho[degrees] = (rho,theta)
			else:
				# hough transform is finding multiple lines, likely because lines are too thick
				# for now, take the average of closely parallel lines
				# later, reevaluate parameters to Canny or HoughLines, or perform preprocessing to thin lines
				if (abs(degreesToRho[degrees][0] - rho) < 10):
					degreesToRho[degrees] = (((degreesToRho[degrees][0] + rho)/2), degreesToRho[degrees][1]) 
			#print('degrees', degrees)
			# a = np.cos(theta)
			# b = np.sin(theta)
			# x0 = a*rho
			# y0 = b*rho
			# x1 = int(x0 + 2000*(-b))
			# y1 = int(y0 + 2000*(a))
			# x2 = int(x0 - 2000*(-b))
			# y2 = int(y0 - 2000*(a))
			# print(x0,y0,x1,y1,x2,y2)
			# print('\n')
			# cv2.line(blank,(x1,y1),(x2,y2),(255,255,255),2)
	print(degreesToRho)
	return degreesToRho
	#parametricCoordinates = []
	# for angle in degreesToRho:
	# 	a = np.cos(degreesToRho[angle][1])
	# 	b = np.sin(degreesToRho[angle][1])
	# 	x0 = a*degreesToRho[angle][0]
	# 	y0 = b*degreesToRho[angle][0]
	# 	x1 = int(x0 + 2000*(-b))
	# 	y1 = int(y0 + 2000*(a))
	# 	x2 = int(x0 - 2000*(-b))
	# 	y2 = int(y0 - 2000*(a))
	# 	print(x0,y0,x1,y1,x2,y2)
	# 	print('\n')
	# 	cv2.line(blank,(x1,y1),(x2,y2),(255,255,255),2)
	# #cv2.imwrite('houghlines3.jpg',img)
	# # print(image.shape)
	# showImage(blank, 'lines')
	#return lines
def determineBounds(image, degreesToRho):
	# in the future, use the image as reference to determine the bounds for each line
	# for now, extend each line to the endge of the image so we can create a cp file
	print(image.shape)
	rows, cols, channels = image.shape
	lines = []
	for angle in degreesToRho:
		a = np.cos(degreesToRho[angle][1])
		b = np.sin(degreesToRho[angle][1])
		x0 = a*degreesToRho[angle][0]
		y0 = b*degreesToRho[angle][0]
		# 3000 is arbitarty, I am trying to shoot every line off the page but might need to revisit for larger images

		x1 = int(x0 + 3000*(-b))
		x1 = max(0,x1)
		x1 = min(cols, x1)

		y1 = int(y0 + 3000*(a))
		y1 = max(0,y1)
		y1 = min(rows, y1)

		x2 = int(x0 - 3000*(-b))
		x2 = max(0,x2)
		x2 = min(cols, x2)

		y2 = int(y0 - 3000*(a))
		y2 = max(0,y2)
		y2 = min(rows, y2)

		line = ((x1,y1), (x2,y2))
		#print(x0,y0,x1,y1,x2,y2)
		#print('\n')
		lines.append(line)
	print(lines)
	return lines

def writeCP(pathToCP, lines):
	with open(pathToCP, 'w+') as f:

		for line in lines:

			cpLine = '1 ' + str(line[0][0]) + ' ' + str(line[0][1]) + ' ' + str(line[1][0]) + ' ' + str(line[1][1])			
			print(cpLine)
			f.write(cpLine)
			f.write('\n')


def showImage(image, title):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Choose an input image')
	parser.add_argument('pathToImage', type=str)
	args = parser.parse_args()

	main(args)