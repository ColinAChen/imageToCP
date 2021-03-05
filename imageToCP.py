import numpy as np
import cv2
import argparse
import math

def main(args):
	# replace transparent background of png with white
	#https://stackoverflow.com/questions/53732747/set-white-background-for-a-png-instead-of-transparency-with-opencv/53737420
	image = cv2.imread(args.pathToImage, cv2.IMREAD_UNCHANGED)
	#make mask of where the transparent bits are
	trans_mask = image[:,:,3] == 0
	#replace areas of transparency with white and not transparent
	image[trans_mask] = [255, 255, 255, 255]
	#new image without alpha channel...
	referenceImage = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

	#image = cv2.imread(args.pathToImage)
	#blank = np.zeros(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape)
	#blank[0:10, 0:10] = 255
	#showImage(blank, '00')
	
	# use hough line transform to determine the location of lines in the input image
	lines = parseImage(args.pathToImage)
	
	# convert the lines from parametric to cartesian coordinates and determine the bounds of each line
	finalLines = determineBounds(referenceImage, lines)
	
	# write these lines to a cp file
	cpFileName = args.pathToImage.split('.')[0] + '.cp'
	writeCP(cpFileName, finalLines)

'''
Read an image, perform preprocessing, and return a dictionary of angles:distances from origin

Arguments:
	pathToImage (str) : path to image to read and convert to .cp
Returns:
	degreesToRho {degrees (float) : [(distance from origin (float), radians from origin (float)) , ...]}
		For a degree we find a line, retrieve a list of distances and angles in radians. 
'''
def parseImage(pathToImage):
	# read image
	#image = cv2.resize(cv2.imread(pathToImage), (400,400))
	# convert to grayscale
	image = cv2.imread(pathToImage)
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#showImage(grayscale, 'grayscale')
	# threshold, send any color to black, all others to white
	ret, grayscale = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY_INV)
	# resize so that everythig is divisible by 2
	# if (grayscale.shape[0] % 2 != 0):
	# 	grayscale = cv2.resize(grayscale, (grayscale.shape[0] * 2,grayscale.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
	#grayscale = thresh
	#showImage(grayscale, 'grayscale')
	# thin lines so that hough transform doesn't pick up lines on both sides of thick lines
	

	grayscale = thinLines(grayscale)
	

	#cv2.imwrite('gray.png', thresh)
	thinned_name = pathToImage.split('.')[0] + '_thinned.png'
	cv2.imwrite(thinned_name, grayscale)
	#showImage(image, 'image')
	#showImage(thinned, 'thinned')
	#grayscale = thinned

	#grayscale = cv2.imread('thinned.png',cv2.IMREAD_GRAYSCALE)
	
	#grayscale = cv2.resize(grayscale, (1000,1000))
	#showImage(grayscale, 'thinned')
	#print(grayscale.shape)
	# for col in range(0,grayscale.shape[1]):
	# 	if (grayscale[100][col] == 0):
	# 		print('black at ', col)
	


	#grayscale = cv2.imread('fish_thinned.png', cv2.IMREAD_GRAYSCALE)
	#showImage(grayscale, 'fish_thinned')
	# perform line detection
	edges = cv2.Canny(grayscale, 50, 150, apertureSize=3)
	#print(edges)
	votes = 20

	lines = cv2.HoughLines(edges,.5,np.pi/32,votes)
	blank = np.zeros(grayscale.shape)
	print(lines)
	print('\n')
	# loopkup a list of distances based on an angle
	degreesToRho = {}

	for line in lines:
		for rho,theta in line:

			#print('rho', rho)
			#print('theta', theta)
			degrees = round((360 * theta) / (2 * math.pi), 3)
			if (degreesToRho.get(degrees) is None):
				degreesToRho[degrees] = [(rho,theta)]
			else:
				# hough transform is finding multiple lines, likely because lines are too thick
				# for now, take the average of closely parallel lines
				# later, reevaluate parameters to Canny or HoughLines, or perform preprocessing to thin lines
				addToList = True
				for i,parametricLine in enumerate(degreesToRho[degrees]):
					#print(parametricLine)
					if (abs(parametricLine[0] - rho) < 10):
						degreesToRho[degrees][i] = (((parametricLine[0] + rho)/2), parametricLine[1])
						addToList = False
				if addToList:
					degreesToRho[degrees].append((rho,theta)) 
			#print('degrees', degrees)
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 2000*(-b))
			y1 = int(y0 + 2000*(a))
			x2 = int(x0 - 2000*(-b))
			y2 = int(y0 - 2000*(a))
			#print(x0,y0,x1,y1,x2,y2)
			#print('\n')
			cv2.line(blank,(x1,y1),(x2,y2),(255,255,255),2)
	showImage(blank, 'blank w/ lines')
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
					(P4 * P6 * P8 == 0) 	and # condition 4 at least one of right, down, left is white
					(P2 * P4 * P6 == 0) 	and # condition 3 at least one of up, right, down is white
					transitions(n) == 1		and # condition 2 there is only one transition from white to black in the list of neighbors
					2*255 <= sum(n) <= 6*255	# condition 1 there are at least two black neighbors and no more than 6
					):
					step1Changes.append((row,col))
		# set all pixels to change from step 1 to white
		#print('step1 changes', step1Changes)
		print('num step 1 changes:', len(step1Changes))
		for row, col in step1Changes: image[row][col] = 255

		# step 2
		step2Changes = []
		for row in range(1,rows-1):
			for col in range(1,cols-1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = getNeighbors(row, col, image)
				if (image[row][col] == 0 	and # condition 0, pixel is black
					(P2 * P6 * P8 == 0) 	and # condition 4 at least one of right, down, left is white
					(P2 * P4 * P8 == 0) 	and # condition 3 at least one of up, right, down is white
					transitions(n) == 1		and # condition 2 There is only one transition from white to black in the list of neighbors
					2* 255 <= sum(n) <= 6 * 255			# condition 1 there are at least two black neighbors and no more than 6
					):
					step2Changes.append((row,col))
		# set all pixels to change from step 1 to white
		#print('step2 changes', step2Changes)
		print('num step 2 changes:', len(step2Changes))
		for row, col in step2Changes: image[row][col] = 255
		#showImage(image, 'in progress')
		## path = 'thinned_' + str(iteration) + '.png'
		## iteration += 1
		## cv2.imwrite(path, image)
	return image

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
				
	return [
		image[row+1][col],
		image[row+1][col+1],
		image[row][col+1],
		image[row-1][col+1],
		image[row-1][col],
		image[row-1][col-1],
		image[row][col-1],
		image[row+1][col-1]
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
	# 0: black, part of object, 1: white, part of background
	n = neighbors + neighbors[0:1]
	return sum((n1,n2) == (0,255) for n1, n2 in zip(n,n[1:]))

'''
Reconstruct the crease pattern by running along the lines we found and referencing the original image

Arguments:
	image (image as numpy array) : original image of crease pattern, ideally in color to determine line type
	degreesToRho (dict) : dictionary mapping degreees to a list of (rho, theta in radians) values, output from parseImage
Returns:
	lines (list of tuples) : [((x1,y1), (x2,y2)), ....] list of lines represented as tuples. Tuples are (start point, end point)
'''
def determineBounds(image, degreesToRho):
	# in the future, use the image as reference to determine the bounds for each line
	# for now, extend each line to the endge of the image so we can create a cp file
	#print(image.shape)
	#image = cv2.imread('fish.png')
	
	#print(image[10][10])
	#print(image[20,30])
	rows, cols, channels = image.shape
	lines = []
	print('\n')
	# keep track of a list of intersections, if a line is close enough to this line, snap it to the intersection by modifying
	# the starting and ending points

	# assume the corners of the square are intersections
	# maybe can also assume havles? The user might be able to specify this?
	# intersections = set()
	# intersections.add((0,0))
	# intersections.add((0, rows-1))
	# intersections.add((cols-1, 0))
	# intersections.add((rows-1, cols-1))
	snapThreshold = rows/100 # might need to change later based on grid size
	intersections = [(0,0), (0, rows-1), (cols-1, 0), (rows-1, cols-1)]
	for angle in degreesToRho:
		for parametricLine in degreesToRho[angle]:
			# parametric line = (rho, theta)
			print('')
			print('angle', angle)
			#print('rho', degreesToRho[angle])
			print('rho', parametricLine[0])
			# a = np.cos(degreesToRho[angle][1])
			# b = np.sin(degreesToRho[angle][1])
			a = np.cos(parametricLine[1])
			b = np.sin(parametricLine[1])
			# x0 = a*degreesToRho[angle][0]
			# y0 = b*degreesToRho[angle][0]
			x0 = a*parametricLine[0]
			y0 = b*parametricLine[0]
			print('x0',x0)
			print('y0',y0)
			### 3000 is arbitarty, I am trying to shoot every line off the page but might need to revisit for larger images
			# 1000 is now just to increase the accuracy of the slope? probably doesn't really matter, it will bring more digits to the left of the decimal
			# x - col, 
			# y - row
			# allow these to be floats, we will round them at the very end
			x1 = float(x0 + 1000*(-b))
			col1 = x1
			#x1 = max(0,x1)
			#x1 = min(cols, x1)

			y1 = float(y0 + 1000*(a))
			row1 = y1
			#y1 = max(0,y1)
			#y1 = min(rows, y1)

			x2 = float(x0 - 1000*(-b))
			col2 = x2
			#x2 = max(0,x2)
			#x2 = min(cols, x2)

			y2 = float(y0 - 1000*(a))
			row2 = y2
			#y2 = max(0,y2)
			#y2 = min(rows, y2)

			##### x1,y1 and x2,y2 are now points on the edge of the square

			# start at one end of the paper
			# iterate through the line along the slope
			vertical = False
			start1 = True
			slope = 0
			if (col2-col1 == 0):
				# no change in col == no change in x
				vertical = True
				print('slope: vertical')
			elif (row2 - row1) > 0:
				# x1 towards x2 is positive
				slope = (row2 - row1) / (col2 - col1)
			else:
				# x2 towards x1 is positive
				# get the slope with increasing column/y
				slope = (row1 - row2) / (col1 - col2) 
				start1 = False
			
			# walk along the line to determine the real bounds
			# in this step, we will reference the original and try to rebuild the crease pattern with intersection accuracy
			# first pass will check if this line is reasonably close to an intersection, if so, we can snap the line to it
			# second pass will reference the original line to determine line bounds and and crease types if the image is in color
			if not vertical:
				print('slope', slope)
				startRow, startCol = start = getStart((row1,col1), slope, rows-1, vertical=vertical)
				offset = (startRow + slope, startCol + 1)
				snappedRow, snappedCol = snap(start, offset, intersections, snapThreshold)
				#snappedRow,snappedCol = startRow, startCol
				print('start',(snappedRow, snappedCol))

				# forward iteration
				startLineSegment = False
				startPoint = (-1,-1)
				endPoint = (-1,-1)
				print('forward range',cols-int(snappedCol))
				print('snapped col:',snappedCol)
				print('snapped row:', snappedRow)
				print('forward iteration')
				finalRow, finalCol = (0,0)
				for colOffset in range(cols-int(snappedCol)):
					checkRow = snappedRow + (slope*colOffset)
					checkCol = snappedCol + colOffset
					finalRow, finalCol = (checkRow, checkCol)
					# if (colOffset == 1):
					# 	print('checkRow', checkRow)
					# 	print('checkCol', checkCol)
					# round to access pixels in the original image
					checkRowInt = int(round(checkRow))
					checkColInt = int(round(checkCol))
					# if (checkRowInt % 37 == 0):

					# 	print('checkRowInt', checkRowInt)
					# 	print('checkColInt', checkColInt)
					if (checkRowInt < 0 or checkRowInt >= rows or checkColInt < 0 or checkColInt >= cols):
						#finalRow, finalCol = (checkRow, checkCol)
						break
					# check if we are one a line
					# might need to check neighbors, depending on how the row and column get rounded
					# if intensity != white, we are on a line
					# in the future, check the color
					# if (checkRowInt == 500):
					# 	print('row, col, intesity', checkRowInt, checkColInt, image[checkRowInt][checkColInt])
					if not colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and not startLineSegment:
						startLineSegment = True
						# snap if this is close enough to an intersection
						offset = (checkRow + slope, checkCol + 1)
						startPoint = snap((checkRow, checkCol), offset, intersections, snapThreshold)
						print('start segment at', startPoint)
					elif colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and startLineSegment:
						startLineSegment = False
						# snap if this is close enough to an intersection
						# check for intersections and update the list of intersections
						line = (startPoint, snap((checkRow, checkCol), startPoint, intersections, snapThreshold))
						if (distance(line[0], line[1]) > 10):

							# add all new intersections this line creates to the list of intersections
							addToIntersections(line, lines, intersections)
							print('end line', line)
							#print(line)
							lines.append(line)
						else:
							print('line found is too short')
				if (startLineSegment):
					# finish unfinished line segments
					line = (startPoint, snap((finalRow, finalCol), startPoint, intersections, snapThreshold))
					if (distance(line[0], line[1]) > 10):

						# add all new intersections this line creates to the list of intersections
						addToIntersections(line, lines, intersections)
						print('end line', line)
						#print(line)
						lines.append(line)
					else:
						print('line found is too short')
			# backward iteration
				print('backward iteration')
				for colOffset in range(int(snappedCol), -1, -1):
					
					checkRow = snappedRow + (slope*colOffset*-1)
					checkCol = snappedCol - colOffset
					finalRow, finalCol = (checkRow, checkCol)
					# if (checkRow < 0 or checkRow > rows or checkCol < 0 or checkCol > cols):
					# 	continue
					# round to access pixels in the original image
					checkRowInt = int(round(checkRow))
					checkColInt = int(round(checkCol))
					# if (checkRowInt % 37 == 0):
						
					# 	print('checkRowInt', checkRowInt)
					# 	print('checkColInt', checkColInt)
					if (checkRowInt < 0 or checkRowInt >= rows or checkColInt < 0 or checkColInt >= cols):
						break
					# check if we are one a line
					# might need to check neighbors, depending on how the row and column get rounded
					# if intensity != white, we are on a line
					# in the future, check the color
					# if (checkRowInt == 500):
					# 	print('row, col, intesity', checkRowInt, checkColInt, image[checkRowInt][checkColInt])
					if not colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and not startLineSegment:
						startLineSegment = True
						# snap if this is close enough to an intersection
						offset = (checkRow + slope, checkCol + 1)
						startPoint = snap((checkRow, checkCol), offset, intersections, snapThreshold)
						print('start segment at', startPoint)
					elif colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and startLineSegment:
						startLineSegment = False
						# snap if this is close enough to an intersection
						# check for intersections and update the list of intersections
						line = (startPoint, snap((checkRow, checkCol), startPoint,intersections, snapThreshold))
						if (distance(line[0], line[1]) > 10):

							# add all new intersections this line creates to the list of intersections
							addToIntersections(line, lines, intersections)
							print('end line', line)
							#print(line)
							lines.append(line)
						else:
							print('line found is too short')
				if (startLineSegment):
					# finish unfinished line segments
					line = (startPoint, snap((finalRow, finalCol), startPoint, intersections, snapThreshold))
					if (distance(line[0], line[1]) > 10):

						# add all new intersections this line creates to the list of intersections
						addToIntersections(line, lines, intersections)
						print('end line', line)
						#print(line)
						lines.append(line)
					else:
						print('line found is too short')
			else:
				# simple case for vertical lines
				# vertical means no change in 
				for row in s

					


				# perform two passes
				# the first pass checks for intersections
			# if row == 0:
			# 	# starting at the bottom
			# else if row == 
			# reference the original image

			# if the pixel there isn't white, there a line here, continue
			# when we find a non-white pixel, start counting
			# when we find a white pixel, stop counting
			# append the line segment to lines

			#line = ((x1,y1), (x2,y2))
			#print(line)
			#print('')

			#print(x0,y0,x1,y1,x2,y2)
			#print('\n')
			#lines.append(line)
	#print(lines)
	print(intersections)
	return lines

def colorEquals(color, target):
	return color[0] == target[0] and color[1] == target[1] and color[2] == target[2]
'''
Snap a point to an existing intersection if it is close enough, essentially, move a line to include an existing intersection if it is wthin a certain threshold

Arguments:
	point (tuple) : (row,col) point on the line, this point will be returned if no existing intersection can be snapped to
	offset (tuple) : (row, col) an different point on the line
	intersections (list) : [(row, col), ...] as list of intersection points
	snapThreshold (number) : must be less than this distance to snap to this point
Return:
	snapPoint (tuple) : (row,col) an existing intersection that is close enough to the line to snap to it, or the original point if no existing intersection is close enough

'''
def snap(point, offset, intersections, snapThreshold):
	print('find snap for point: ', point)
	print('using offset:', offset)
	# calculate the distance between every intersection and this line
	distanceToIntersections = [distanceLineToPoint(point, offset, intersection) for intersection in intersections]
	#distanceToPoints = [distance(point, intersection) for intersection in intersections]
	#print('distances', distanceToIntersections)
	# get the shortest distance
	minDistance = min(distanceToIntersections)
	if (minDistance < snapThreshold):
		# if the minimum distance is withing snapThreshold away,
		# assume the intersections is part of the line and return the intersection

		# also assume that the closest euclidean point is the one we want to snap to
		# this won't affect snapping to a point for iteration
		# but will help with snapping to final points after iteration.
		minIndicies = list(filter(lambda x: distanceToIntersections[x] == minDistance, range(len(distanceToIntersections))))
		#print(minIndicies)
		#minIndex = distanceToIntersections.index(minDistance)
		#print('minIndex', minIndex)
		#print('snap to ', intersections[minIndex])
		minDistance = 999999999
		minIndex = 0
		for index in minIndicies:
			checkDistance = distance(point, intersections[index])
			if (checkDistance < minDistance):
				minDistance = checkDistance
				minIndex = index
		#print('minDistance', minDistance)
		#print('minIndex', index)
		
		if minDistance < snapThreshold:
			print('snap to ', intersections[minIndex])
			return intersections[minIndex]
		else:
			#print('snap to ', intersections[minIndex])
			return point
	else:
		# otherwise, return the point
		return point

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
Calculate the euclidena distance betwen two points
Arguments:
	point1 (tuple) : (row, col) point 1
	point2 (tuple) : (row, col) point 2
Returns:
	distance (float) : distance between two points. 
'''
def distance(point1, point2):
	return math.sqrt( ((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1])**2) )

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
		return True
	return False
	# Special Cases 

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
def addToIntersections(newLine, lines, intersections):
	for line in lines:
		if lineSegmentIntsersect(newLine, line):
			intersection = findLineSegmentIntersect(newLine, line)
			intersections.append(intersection)
def findLineSegmentIntersect(line1, line2):
	#https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
	print(line1)
	print(line2)
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y
'''
Find the leftmost intersection between this line and the square

Arguments: 
	point (tuple) : (row, col) point on the line
	slope (float) : slope of the line
	length (int)  : length of square
Return:
	squareStart (tuple) : (row, col) leftmost point on the square
'''
def getStart(point, slope, length, vertical=False):
	# slope = change in row/change in column
	row,col = point
	print('finding start for', point)
	if (vertical):
		# this is a vertical line, return the column intersection with row 0
		return (0, col)
	elif (slope == 0):
		# this is a horizontal line, return the row intersection with column 0
		return (row,0)
	elif (col == 0):
		# point is on the left edge, this is an optimal starting point
		return point
	else:
		### GENERAL ALGORITHM ###
		# check the intersection with the left edge
		# if the row is on the left edge, return that point
		# if the row is above the left edge, return the intersection with the top edge
		# if the row is below the left edge, return the intersection with the bottom edge

		# conceptually, -1 will either swap col in the case the col < 0 or swap slope inthe case that col > 0
		leftRow = (col * slope * -1) + row
		print('left row', leftRow)
		if (0 <= leftRow <= length):
			# intersection with the left edge is in bounds
			return (leftRow, 0)
		elif (leftRow < 0):
			# intersection with bottom edge
			return (0, col - (row/slope))
		else:
			# intersection with top edgess
			return (length, col - ((length-row) / slope))


'''
Write a .cp file with a given list of lines

Arguments:
	pathToCP (str) : path to the cp file
	lines (list) : list of lines, output from determineBounds
Returns:
	No return but will create a file at path of pathToCP in the .cp file format
'''
def writeCP(pathToCP, lines):
	print('writing to ', pathToCP)
	with open(pathToCP, 'w+') as f:

		for line in lines:

			cpLine = '1 ' + str(line[0][0]) + ' ' + str(line[0][1]) + ' ' + str(line[1][0]) + ' ' + str(line[1][1])			
			print(cpLine)
			f.write(cpLine)
			f.write('\n')

'''
Helper function to show and destroy images

Arguments:
	image (image as numpy array) : image to display
	title (str) : title of display window
Returns:
	None
'''
def showImage(image, title):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Choose an input image')
	parser.add_argument('pathToImage', type=str)
	args = parser.parse_args()

	main(args)