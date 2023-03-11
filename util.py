import numpy as np
import cv2
import argparse
import math
import os
import sys
import time
import random

SUPRESS = True

BLUE = (255,0,0)
RED = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)
'''
Get the eight neighbors of a pixel in the order specified by the Zhange-Suen line thinning algorithm

Arguments:
	row (int) : row of target pixel
	col (int) : column of target pixel
	image (image as numpy array) : image target pixel and neighbors are from

Returns:
	neighbors (list) : [intesity of up, up-right, right, down-right, down, down-left, left, up-left]
'''
def getNeighbors(row, col, image):
				
	# return [
	# 	image[row+1][col],
	# 	image[row+1][col+1],
	# 	image[row][col+1],
	# 	image[row-1][col+1],
	# 	image[row-1][col],
	# 	image[row-1][col-1],
	# 	image[row][col-1],
	# 	image[row+1][col-1]
	# 	]
	return [
		image[row-1][col],
		image[row-1][col+1],
		image[row][col+1],
		image[row+1][col+1],
		image[row+1][col],
		image[row+1][col-1],
		image[row][col-1],
		image[row-1][col-1]
		]

'''
Get the number of transitions from black to white, might actually be reversed but doesn't seem to affect the output

Arguments:
	neighbors (list) : list of neighbors, pass output from getNeighbors to ensure correct order
Returns:
	transitions (int) : number of transitions from black to white in the neighbor list
'''
def transitions(neighbors):
	# how many times do pixels in the neighbor list swap from white to black?
	# 0: black, part of object, 255: white, part of background
	n = neighbors + neighbors[0:1]
	return sum((n1,n2) == (0,255) for n1, n2 in zip(n,n[1:]))

'''
Thin lines found in an image
White (255) : background
Black (0)	: foreground/line color
	Invert your image if this is not the case

Arguments:
	image (image as numpy array) : image to thin lines in
Returns:
	image (image as numpy array) : image is modified in place but return it anyway 
'''
def thinLines(image):
	# using the Zhang-Suen image line thinning algorithm
	# https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm#Python
	#print(image.shape	)
	rows, cols = image.shape
	# avoid the first and last pixels to ensure pixels always have 8 neighbors
	#numChanges = 0
	step1Changes = [(-1,-1)]
	step2Changes = [(-1,-1)]
	firstPass = True
	iteration = 1
	while (step1Changes or step2Changes):
		# step 1
		step1Changes = []
		for row in range(1,rows-1):
			for col in range(1,cols-1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = getNeighbors(row, col, image)
				#if (row > 500 and row < 510 and col == 20):
					#print('row,col',row,col)
					#print(n)
					#print(transitions(n))
					#print(sum(n))
					#print('\n')
				if (image[row][col] == 0 	and # condition 0, pixel is black
					(float(P4) + P6 + P8 > 0) 	and # condition 4 at least one of right, down, left is white
					(float(P2) + P4 + P6 > 0) 	and # condition 3 at least one of up, right, down is white
					transitions(n) == 1		and # condition 2 there is only one transition from white to black in the list of neighbors
					2*255 <= sum(n) <= 6*255	# condition 1 there are at least two black neighbors and no more than 6
					):
					step1Changes.append((row,col))
		# set all pixels to change from step 1 to white
		#print('step1 changes', step1Changes)
		print('num step 1 changes:', len(step1Changes))
		for row, col in step1Changes: image[row][col] = 255
		showImage(image, 'step1')
		# step 2
		step2Changes = []
		for row in range(1,rows-1):
			for col in range(1,cols-1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = getNeighbors(row, col, image)
				if (image[row][col] == 0 	and # condition 0, pixel is black
					(float(P2) + P6 + P8 > 0) 	and # condition 4 at least one of right, down, left is white
					(float(P2) + P4 + P8 > 0) 	and # condition 3 at least one of up, right, down is white
					transitions(n) == 1		and # condition 2 There is only one transition from white to black in the list of neighbors
					2* 255 <= sum(n) <= 6 * 255			# condition 1 there are at least two black neighbors and no more than 6
					):
					step2Changes.append((row,col))
		# set all pixels to change from step 1 to white
		#print('step2 changes', step2Changes)
		print('num step 2 changes:', len(step2Changes))
		for row, col in step2Changes: image[row][col] = 255
		showImage(image, 'step 2')
		## path = 'thinned_' + str(iteration) + '.png'
		## iteration += 1
		## cv2.imwrie(path, image)
	return image


'''
Helper debugging displays
'''
def showLine(line,image, lineType=1):
	c1,r1 = line[0]
	c2,r2 = line[1]
	lineColor = WHITE
	if lineType==2:
		lineColor = RED
	if lineType==3:
		lineColor = BLUE
	cv2.line(image,(int(r1), int(c1)),(int(r2), int(c2)),lineColor,5)
	image = cv2.resize(image, (400,400))
	showImage(image,'line added')
	#image = np.zeros(image.shape)
def highlightPoint(point,image):
	drawImage = image.copy()
	cv2.circle(drawImage, point, 20, (0,255,0))
	drawImage = cv2.resize(drawImage, (400,400))
	showImage(drawImage,'highlight')
'''
Check if two colors are equal by checking each channel individually
Only used because tuple and numpy channel array are slightly different
'''
def colorEquals(color, target):
	return color[0] == target[0] and color[1] == target[1] and color[2] == target[2]

'''
Maybe can change to checking RED and BLUE and BLACK instead?
'''
def getLineType(color):
	# return 1 for none
	# return 2 for mountain
	# return 3 for valley
	if (color[0] > color[1] and color[0] > color[2]):
		return 3
	elif (color[2] > color[0] and color[2] > color[1]):
		return 2
	else:
		return 1


'''
Calculate the euclidean distance betwen two points
Arguments:
	point1 (tuple) : (row, col) point 1
	point2 (tuple) : (row, col) point 2
Returns:
	distance (float) : distance between two points. 
'''
def distance(point1, point2):
	if (point1 is None or point2 is None):
		return 999999999
	return math.sqrt(sum([(x1 - x2)**2 for x1,x2 in zip(point1, point2)]))
	#return math.sqrt( ((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1])**2) )

'''
Helper function to show and destroy images

Arguments:
	image (image as numpy array) : image to display
	title (str) : title of display window
Returns:
	None
'''
def showImage(image, title='image', scale=False):
	if SUPRESS:
		return
	if scale:
		copy = cv2.resize(image,(750,750))
		cv2.imshow(title, copy)
	else:
		cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



'''
Write a .cp file with a given list of lines

Arguments:
	pathToCP (str) : path to the cp file
	lines (list) : list of lines, output from determineBounds
Returns:
	No return but will create a file at path of pathToCP in the .cp file format
'''
def writeCP(pathToCP, lines, show=False):
	print('\nwriting to ', pathToCP)

	with open(pathToCP, 'w+') as f:
		for line in lines:
			#print(line)
			cpLine = str(lines[line]) + ' ' + str(line[0][0]) + ' ' + str(line[0][1]) + ' ' + str(line[1][0]) + ' ' + str(line[1][1])			
			print(cpLine)
			f.write(cpLine)
			f.write('\n')
	if (show):
		# find the bounds
		# probably only need to check the max, min should always be zero
		minVal, maxVal = 99999, -1

		for line in lines:
			checkMin = min(line[0][0], line[0][1], line[1][0], line[1][1])
			checkMax = max(line[0][0], line[0][1], line[1][0], line[1][1])
			if (checkMin < minVal):
				minVal = checkMin
			if (checkMax > maxVal):
				maxVal = checkMax
		# create a square to draw lines on
		maxVal = int(maxVal)
		square = np.zeros((maxVal, maxVal))
		print(square.shape)
		for line in lines:
			cv2.line(square,(int(line[0][0]), int(line[0][1])),(int(line[1][0]), int(line[1][1])),(255,255,255),2)
		showImage(square)
'''
Determine if two line segments intersect
https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
Arguments:
	line1 : ((x1, y1), (x2, y2)), line segment 1
	line2 : ((x1, y1), (x2, y2)), line segment 2
Returns:
	True if the line segments intersect, False otherwise

'''
def lineSegmentIntsersect(line1, line2):
	p1,q1 = line1
	p2,q2 = line2

	# Find the 4 orientations required for  
	# the general and special cases 
	o1 = orientation(p1, q1, p2) 
	o2 = orientation(p1, q1, q2) 
	o3 = orientation(p2, q2, p1) 
	o4 = orientation(p2, q2, q1) 
	# General case 
	if ((o1 != o2) and (o3 != o4)): 
		#print(line1,'intersects with',line2)
		return True
	return False
	# Special Cases 
	# Ignore specials bases because we should never encounter parallel line, colinear lines

	# # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
	# if ((o1 == 0) and onSegment(p1, p2, q1)): 
	# 	return True

	# # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
	# if ((o2 == 0) and onSegment(p1, q2, q1)): 
	# 	return True

	# # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
	# if ((o3 == 0) and onSegment(p2, p1, q2)): 
	# 	return True

	# # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
	# if ((o4 == 0) and onSegment(p2, q1, q2)): 
	# 	return True

	# # If none of the cases 
	# return False
def onSegment(p,q,r):
	px,py = p
	qx,qy = q
	rx,ry = r
	return qx <= max(px, rx) and qx >= min(px, rx) and qy <= max(py, ry) and qy >= min(py, ry)

def orientation(p,q,r):
	# 0, colinear, slopes are the same
	# 1 clockwise points
	# 2 counter clockwise
	px,py = p
	qx,qy = q
	rx,ry = r
	value = (float(qy-py) * (rx-qx)) - (float(qx-px) * (ry-qy)) 
	if (value > 0):
		return 1
	elif (value < 0):
		return 2
	else:
		return 0

'''
newLine (float start row, float start col) : new line to check for intersections with existing lines
lines {(float start row, float start col) : lineType, ....} : dict of existing lines, assume they are intersection free (form a planar graph) with associated lineType
intersections set{(float intersect row, float intersect col)} : set of intersections, use to check if there is nearby intersection so the new intersection shouldn't be registered
lineType (int 1,2, or 3) : line type of the newLine, used to create new lines that meet at the intersection
rejectThreshold (number) : distance to an existing intersection to which a new intersection should not be added to the set of intersections
'''
def addToIntersections(newLine, lines, intersections, lineType, rejectThreshold=10):
	print('registering intersections for',newLine)
	# add line endpoints as possible intersections for future lines
	# the nature of adding lines in a weird order means that they will be future intersections
	# assuming the model is flat foldable and the lines are inside the square


	# we need to create new lines that all meet at the intersection

	# need to reject points that are very close to an existing intersection
	if (newLine[0] not in intersections):
		intersections.append(newLine[0])
	if (newLine[1] not in intersections):
		intersections.append(newLine[1])
	addLines = []
	removeLines = []
	recheckList = []
	# check each line
	# if there is an intersection
	# check whether or not to add this intersection to the set
	# recrate lines that meet at the intersection, we will need to check the new segments for intersections as well
	for line in lines:
		#print('check line', line)
		if lineSegmentIntsersect(newLine, line):

			intersection = findLineSegmentIntersect(newLine, line)
			#print(newLine, 'intersects with', line, 'at', intersection)
			if intersection is not None and min([distance(intersection, checkIntersection) for checkIntersection in intersections]) > rejectThreshold:
			#if (intersection not in intersections):
				intersections.append(intersection)

			# create new lines that meet at the intersection
			newLine1 = (newLine[0], intersection)
			newLine2 = (intersection, newLine[1])
			newLine3 = (line[0], intersection)
			newLine4 = (intersection, line[1])
			checkNewLines = (newLine1, newLine2, newLine3, newLine4)
			for checkLine in checkNewLines:
				# ensure that each line is long enough
				# arbitrary, might want to base this off of the image size later
				if (distance(checkLine[0], checkLine[1]) > 5):
					addLine = (checkLine, lineType)
					addLines.append(addLines)


'''
Find the intersection between two line segments
Assumes they intersect but that might be redundant, can likely be optimized

Arguments:
	line1 (tuple) : ((x1,y1), (x2,y2)) line segment 1
	line2 (tuple) : ((x1,y1), (x2,y2)) line segment 2
Returns:
	intersecction (tuple) : (x,y) intersection point
	Can also raise Exception but this should never happen in its current use cases
'''
def findLineSegmentIntersect(line1, line2):
	#https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
	#print(line1)
	#print(line2)
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		#print('LINES DO NOT INTERSECT')
		return None
		#raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y
'''
Distance from a point to a line
https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

Arguments:
	p1 (tuple) : (row,col) point on line
	p1 (tuple) : (row,col) a different point on line
	target (tuple) : (row,col) point to caluclate distance to
Return:
	distance : distance from point to the line
'''
def distanceLineToPoint(p1, p2 ,target):
	x1,y1 = p1
	x2,y2 = p2
	x0,y0 = target
	if (x1 == x2):
		# horizontal line, travels along a row
		# return y displacement (column difference)
		#print('horizontal line')
		return abs(x1 - x0)
	if (y1 == y2):
		# vertical line, travels along the same column
		# return x displacement (row difference)
		#print('vertical line')
		return abs(y1 - y0)
	#((x2 - x1) * (y1 - y0)) * ((x1 - x0) * (y2 - y1))
	#numerator = abs((a * x) + (b * y) + c)
	numerator = abs(((x2 - x1) * (y1 - y0)) - ((x1 - x0) * (y2 - y1)))
	#denominator = math.sqrt((a*a) + (b*b))
	denominator = distance(p1,p2)
	return numerator/denominator

'''
Use OS to create a path agnostic of OS

'''
def createPath(path):

	pass


# thinninning lines then expanding to create 3 pixel lines > just using the lines?
# let's revist to preserve lines, EDIT this is probably unecessary, I think the current paradigm is sufficient
# like FB summer 2021 coding interview
# this currently takes a long time, maybe can optimize later
def expand(image):
	#pass
	# editSet = []
	# for row in image:
	# 	for pixel in row:
	# 		if ()
	rows, cols, channels = image.shape
	retImage = np.zeros(image.shape)
	neighborAccess = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (0,1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (0,1)]
	for r, row in enumerate(image):
		if (r == 0 or r == rows-1):
			continue
		for c, pixel in enumerate(row):
			if (c == 0 or c == cols-1):
				continue
			if (colorEquals(pixel, WHITE)):
				# only need to edit white pixels
				# determine the most populous neighbor
				redCount = 0
				blueCount = 0
				for add in neighborAccess:
					if colorEquals(image[r + add[0]][c + add[1]], RED):
						redCount+=1
					elif colorEquals(image[r + add[0]][c + add[1]], BLUE):
						blueCount+=1
				if (redCount >= blueCount and redCount > 0):
					# arbitrarily set a pixel to red if it is a tie
					retImage[r][c] = RED
				elif blueCount > 0:
					retImage[r][c] = BLUE
				else:
					retImage[r][c] = image[r][c]
			else:
				# keep black, blue, and red pixels
				retImage[r][c] = image[r][c]
	showImage(retImage, title='expand')
	return retImage

def checkNeighbor(row, col, image):
	pass
	# checkSet = 
	# if (row == 0):

'''
set all pixels to
black (0,0,0)
white(255,255,255)
red (0,0,255)
blue(255,0,0)

Look at pixel value and maybe neighbors
'''	
def deblur(image):
	colors = [
	[0,0,0],
	[255,255,255],
	[0,0,255],
	[255,0,0]
	]
	for r, row in enumerate(image):
		for c, pixel in enumerate(row):
			# determine the closest color
			bestMatch = colors[0]
			minDistance = 99999
			if (colorEquals(pixel, colors[0]) or colorEquals(pixel, colors[1]) or colorEquals(pixel, colors[2]) or colorEquals(pixel, colors[3])):
				# skip if this pixel is already one of the acceptable colors
				continue
			for color in colors:
				if (distance(color, pixel) < minDistance):
					minDistance = distance(color, pixel)
					bestMatch = color
			# set the pixel to that color
			image[r][c] = bestMatch
	#print(image.shape)
	#showImage(image)

	# should be modifying in place but need to return for the way I called it?
	return image

'''
dfsLines, use depth first search to approximate line starts and slopes, use this an alternative to hough line transform

Arguments: 
	image (image as numpy array) : copy of binary grayscale, line thinned image of crease. Idealy, lines will have at thickness of 1 pixel

Returns: 
	List of line endppoints
'''
def dfsLines(image):
	'''
	for pixel in image:

		if (pixel is black (on a line)):
			add pixel to stack
			dfsHelper(on black neighbors, these should all be unique lines themselves) -> last pixel in this line
			set pixel to white as we go
			add start, end to list
			return list of points
		

	'''
	visitedSet = set()
	queue = []
	# explore every pixel
	for row, imageRow in enumerate(image):
		for col, pixel in enumerate(imageRow):
			# start searching once we find a 
			if pixel == WHTIE:
				pass
def dfsRecursiveHelper(row, col, image):
	pass



'''
find the intersections in a crease pattern to create a complete point set

Arguments
	image (numpy array as image) : image of crease pattern, preferably cropped

Returns
	intersections [(float, float)] : list of intersection points
'''
def findIntersections(image):
	'''
	Some ideas to try
	sliding window, intersections should have more color than non intersections
	'''
	# sliding window can be based on image size, hardcode for now
	windowSize = 5

	# pad the image to include corners
	gray = image
	if len(image[0][0]) > 1:
		gray = cv2.cvtColor(image, cv2.COLOR_BGRTOGRAY)
	# set lines to white and lines to black
	# for now assume that lines are black and background is white
	# we can detect this later, but probably should detect this upstream instead
	gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	showImage(gray)

	# high conv score means more lines
	# threshold so all lines types contribute equally
	
	rows, cols = gray.shape
	conv = np.zeros((rows + (2*windowSize), cols + (2*windowSize)))



