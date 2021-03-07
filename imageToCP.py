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
	#testColor = referenceImage.copy()

	#testColor[0:100,0:100] = [0,0,255]
	#BLUE = [255,0,0]
	#RED = [0,0,255]
	#showImage(testColor, 'testColor')
	#image = cv2.imread(args.pathToImage)
	#blank = np.zeros(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape)
	#blank[0:10, 0:10] = 255
	#showImage(blank, '00')
	


	
	# use hough line transform to determine the location of lines in the input image
	if (args.pathToThinned):

		lines = parseImage(args.pathToThinned)
	else:
		lines = parseImage(args.pathToImage, needThinLines=True)
	
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
def parseImage(pathToImage, needThinLines=False):
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

	if (needThinLines):

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
			#showImage(blank, 'line')
			#blank = np.zeros(blank.shape)
	#showImage(blank, 'blank w/ lines')
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
					(float(P4) * P6 * P8 == 0) 	and # condition 4 at least one of right, down, left is white
					(float(P2) * P4 * P6 == 0) 	and # condition 3 at least one of up, right, down is white
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
					(float(P2) * P6 * P8 == 0) 	and # condition 4 at least one of right, down, left is white
					(float(P2) * P4 * P8 == 0) 	and # condition 3 at least one of up, right, down is white
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
	blank = np.zeros(image.shape)
	
	#lines = [] # consider adding the square edges as lines to create intersections with?
	lines = {} # {line : line type} # need to keep track of line type 
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
	snapThreshold = 100 #rows/100 # might need to change later based on grid size
	corners = [(0,0), (0, rows-1), (cols-1, 0), (rows-1, cols-1)]
	midpoints = [(0,(cols-1)/2), ((rows-1)/2, 0), ((rows-1)/2, cols-1), (rows-1,(cols-1)/2)] # can add grid lines here later
	#intersections = [(0,0), (0, rows-1), (cols-1, 0), (rows-1, cols-1)] # consider adding grid lines as bounds here?
	intersections = corners# + midpoints
	# gridSize = 4
	# for i in range(gridSize+1):
	# 	for j in range(gridSize+1):
	# 		if ((i * (rows-1)/gridSize, j * (cols-1)/gridSize) not in intersections):
	# 			intersections.append((i * (rows-1)/gridSize, j * (cols-1)/gridSize))

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
			# showLine(((row1, col1), (row2,col2)), blank)
			# blank = np.zeros(blank.shape)
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
				# forward pass
				referenceImage((row1, col1), slope, intersections, lines, image, snapThreshold, forward=True, vertical=False)
				# backward pass
				referenceImage((row1, col1), slope, intersections, lines, image, snapThreshold, forward=False, vertical=False)
			else:
				# vertical line, slope doesn't matter
				referenceImage((row1, col1), -1, intersections, lines, image, snapThreshold, forward=True, vertical=True)
	#print(lines)
	print('intersections',intersections)
	# add the corners to lines to create a square
	# top edge
	lines[(corners[0], corners[1])] = 1
	# left edge
	lines[(corners[0], corners[2])] = 1
	# right edge
	lines[(corners[2], corners[3])] = 1
	# bottom edge
	lines[(corners[1], corners[3])] = 1
	return lines
'''
Code is repeated three times
1 forward iteration from start point
2 backward iteration from start ppint
3 vertical line edge case to handle no change in column
'''
def referenceImage(start, slope, intersections, lines, image, snapThreshold, forward=True, vertical=False):
	'''
	In this step, we take the lines we calculated in parseImage, and attempt to create a crease pattern by referecing the original image.
	The lines we receive in parseImage have no endpoints, so we must figure out the endpoints in this step.
	The fact that this is an image means that the pixels values we observe can only approximate the actual points these crease make.
	We will recreate the crease pattern by using some knowledge of the crease pattern (for example, a grid size). 

	To recreate the lines, iterate over all the pixels this line passes through on the square
	We determine the bounds with a simple state machine

	'''
	RED = (0,0,255)# Red = (0, 0, 255)
	BLUE = (255,0,0)# Blue = (255, 0, 0)

	colorThreshold = 60 # totally arbitary, not sure how this will play out for shorter line segments
	rows,cols,channels = image.shape
	row1,col1 = start
	blank = np.zeros(image.shape)
	print('slope', slope)
	startRow, startCol = start = getStart((row1,col1), slope, rows-1, vertical=vertical)
	offset = (startRow + slope, startCol + 1)
	snappedStart, snappedStartOffset = snap(start, offset, intersections, snapThreshold)
	snappedRow, snappedCol = snappedStart
	#snappedRow,snappedCol = startRow, startCol
	print('start iteration',(snappedRow, snappedCol))

	# loop is the same
	# define loop bounds
	bounds = range(1)
	if (vertical):

		# vertical line
		bounds = range(rows)
		print ('vertical bounds', bounds)
	elif(forward):
		# forward iteration
		bounds = range(cols-int(snappedCol))
		print('forward pass bounds', bounds)
	else:
		# backward iteration
		bounds = range(int(snappedCol), -1, -1)
		print('backward pass bounds', bounds)
	
	# variables for line recreation state machine
	startLineSegment = False
	startPoint = (-1,-1)
	endPoint = (-1,-1)
	# 
	rollingAverageColor = [0,0,0] #currently getting rounded, not sure how that is affecting the accuracy
	linePixels = 0
	#print('forward range',cols-int(snappedCol))
	#print('snapped col:',snappedCol)
	#print('snapped row:', snappedRow)
	#print('forward iteration')
	finalRow, finalCol = (0,0)
	snapStart = False # keep track of whether or not we snapped the starting point
	for offset in bounds:
		checkRow = snappedRow + (slope*offset)
		checkCol = snappedCol + offset
		if (vertical):
			# handle the vertical line caes seperately
			checkRow = offset
			checkCol = snappedCol
		# keep track of the last point we visit in case we have an unfinished line
		finalRow, finalCol = (checkRow, checkCol)
		# round to access pixels in the original image
		checkRowInt = int(round(checkRow))
		checkColInt = int(round(checkCol))
		if (checkRowInt < 0 or checkRowInt >= rows or checkColInt < 0 or checkColInt >= cols):
			#finalRow, finalCol = (checkRow, checkCol)
			# protect image from referencing points out of bounds
			break
		# check if we are one a line
		# might need to check neighbors, depending on how the row and column get rounded
		# if intensity != white, we are on a line
		# in the future, check the color
		# if (checkRowInt == 500):
		# 	print('row, col, intesity', checkRowInt, checkColInt, image[checkRowInt][checkColInt])
		if not colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and not startLineSegment:
			print('start line at',checkRow, ',',checkCol,'with intensity', image[checkRowInt][checkColInt])
			startLineSegment = True
			# snap if this is close enough to an intersection
			offset = (checkRow + slope, checkCol + 1)
			startPoint, startOffset = snap((checkRow, checkCol), offset, intersections, snapThreshold)
			print('start segment at', startPoint)
			# keep track of the color as a rolling average
			# this will allow us to determine the color of lines even if they are occluded by differently colored lines
			rollingAverageColor = image[checkRowInt][checkColInt]
			linePixels = 1
			#highlightPoint((checkColInt,checkRowInt),image.copy())
		elif not colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and startLineSegment:
			#print('rollingAverageColor', rollingAverageColor)
			#print('referencePixel',image[checkRowInt][checkColInt])
			# we are currently on an unfinished line segment, just keep track of the color
			# keep track of the rolling average pixel
			index = 0
			for averagePixel, referencePixel in zip(rollingAverageColor, image[checkRowInt][checkColInt]):
				rollingAverageColor[index] = ((averagePixel * linePixels) + referencePixel) / (linePixels + 1)
				index += 1
			linePixels += 1
		elif colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and startLineSegment:
			# we have started a line and just encountered a white pixel
			# this means the line we are on has finished
			startLineSegment = False
			# snap if this is close enough to an intersection
			# check for intersections and update the list of intersections
			endPoint, endOffset = snap((checkRow, checkCol), startPoint,intersections, snapThreshold)
			line = (startPoint, endPoint)
			if (distance(line[0], line[1]) > 10):
				# determine the line type
				# opencv pixel intensities are returned as BlueGreenRed
				print('rollingAverageColor for line:',rollingAverageColor)
				lineType = 1
				if (distance(rollingAverageColor, RED) < colorThreshold):
					# red line indicates mountain fold
					# .cp -> mountain fold is 2
					print('mountain fold')
					lineType = 2
				elif (distance(rollingAverageColor, BLUE) < colorThreshold):
					# blue line indiciates valley fold
					# .cp -> valley fold is 3
					print('valley fold')
					lineType = 3
				# add all new intersections this line creates to the list of intersections
				#addToIntersections(line, lines, intersections)
				addToIntersections(line, lines.keys(), intersections)
				print('end line', line)
				#showLine(line, blank.copy())
				#blank = np.zeros(blank.shape)
				#print(line)
				#lines.append(line)
				lines[line] = lineType
			else:
				print('line found is too short')

	if (startLineSegment):
		startLineSegment = False
		print('need to finish line segment')
		# finish unfinished line segments
		endPoint, endOffset = snap((finalRow, finalCol), startPoint,intersections, snapThreshold)
		line = (startPoint, endPoint)
		if (distance(line[0], line[1]) > 10):
			# determine the line type
			# opencv pixel intensities are returned as BlueGreenRed
			print('rollingAverageColor for line:',rollingAverageColor)
			lineType = 1
			if (distance(rollingAverageColor, RED) < colorThreshold):
				# red line indicates mountain fold
				# .cp -> mountain fold is 2
				print('mountain fold')
				lineType = 2
			elif (distance(rollingAverageColor, BLUE) < colorThreshold):
				# blue line indiciates valley fold
				# .cp -> valley fold is 3
				print('valley fold')
				lineType = 3
			# add all new intersections this line creates to the list of intersections
			addToIntersections(line, lines.keys(), intersections)
			print('end line', line)
			#showLine(line, blank.copy())
			#blank = np.zeros(blank.shape)
			#print(line)
			#lines.append(line)
			lines[line] = lineType
		else:
			print('line found is too short')

def showLine(line,image):
	c1,r1 = line[0]
	c2,r2 = line[1]
	
	cv2.line(image,(int(r1), int(c1)),(int(r2), int(c2)),(255,255,255),10)
	image = cv2.resize(image, (400,400))
	showImage(image,'line added')
	#image = np.zeros(image.shape)
def highlightPoint(point,image):
	drawImage = image.copy()
	cv2.circle(drawImage, point, 100, (0,255,0))
	drawImage = cv2.resize(drawImage, (400,400))
	showImage(drawImage,'highlight')
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
	offset (tuple) : (delta row, delta col) offset from point to snap, used to shift the starting point if applicable

'''
def snap(point, offset, intersections, snapThreshold):
	row,col = point
	### also return the snap offset so we can offset the whole line?
	print('find snap for point: ', point)
	print('using offset:', offset)
	# calculate the distance between every intersection and this line
	distanceToIntersections = [round(distanceLineToPoint(point, offset, intersection),4) for intersection in intersections]
	#distanceToPoints = [distance(point, intersection) for intersection in intersections]
	#print('distances', distanceToIntersections)
	# get the shortest distance
	#minDistance = min(distanceToIntersections)
	#print(distanceToIntersections)
	#if (minDistance < snapThreshold):
	# if the minimum distance is withing snapThreshold away,
	# assume the intersections is part of the line and return the intersection

	# also assume that the closest euclidean point is the one we want to snap to
	# this won't affect snapping to a point for iteration
	# but will help with snapping to final points after iteration.
	# indicies of intersections that are within snapping distance of point
	minIndicies = list(filter(lambda x: distanceToIntersections[x] < snapThreshold, range(len(distanceToIntersections))))
	#print(minIndicies)
	#minIndex = distanceToIntersections.index(minDistance)
	#print('minIndex', minIndex)
	#print('snap to ', intersections[minIndex])
	minDistance = 999999999
	minIndex = 0
	for index in minIndicies:
		#print(intersections[index])
		checkDistance = distance(point, intersections[index])
		if (checkDistance < minDistance):
			minDistance = checkDistance
			minIndex = index
	#print('minDistance', minDistance)
	#print('minIndex', index)
	
	if minDistance < snapThreshold:
		snapRow, snapCol = snapTo = intersections[minIndex]
		print('snap to ', snapTo)
		return intersections[minIndex], (snapRow-row, snapCol-col)
	# elif (len(:
	# 	#print('snap to ', intersections[minIndex])
	# 	print('point on line but not close enough, returning point')
	# 	return point
	else:
		# otherwise, return the point
		print('no points near line')
		return point, (0,0)

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
Calculate the euclidean distance betwen two points
Arguments:
	point1 (tuple) : (row, col) point 1
	point2 (tuple) : (row, col) point 2
Returns:
	distance (float) : distance between two points. 
'''
def distance(point1, point2):
	return math.sqrt(sum([(x1 - x2)**2 for x1,x2 in zip(point1, point2)]))
	#return math.sqrt( ((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1])**2) )

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
	print('registering intersections for',newLine)
	# add line endpoints as possible intersections for future lines
	# the nature of adding lines in a weird order means that they will be future intersections
	# assuming the model is flat foldable an the lines are inside the square
	if (newLine[0] not in intersections):
		intersections.append(newLine[0])
	if (newLine[1] not in intersections):
		intersections.append(newLine[1])
	for line in lines:
		#print('check line', line)
		if lineSegmentIntsersect(newLine, line):

			intersection = findLineSegmentIntersect(newLine, line)
			print(newLine, 'intersects with', line, 'at', intersection)
			if (intersection not in intersections):

				intersections.append(intersection)

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
		# leeway should be taken care of by snap
		#if (distance())
		if (-10 <= leftRow <= length+10):
			# intersection with the left edge is in bounds
			return (leftRow, 0)
		elif (leftRow < 0):
			# intersection with bottom edge
			return (0, col - (row/slope))
		else:
			# intersection with top edgess
			return (length, col + ((length-row) / slope))


'''
Write a .cp file with a given list of lines

Arguments:
	pathToCP (str) : path to the cp file
	lines (list) : list of lines, output from determineBounds
Returns:
	No return but will create a file at path of pathToCP in the .cp file format
'''
def writeCP(pathToCP, lines):
	print('\nwriting to ', pathToCP)
	with open(pathToCP, 'w+') as f:
		for line in lines:
			cpLine = str(lines[line]) + ' ' + str(line[0][0]) + ' ' + str(line[0][1]) + ' ' + str(line[1][0]) + ' ' + str(line[1][1])			
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
	parser.add_argument('--pathToThinned','-p', type=str, default='')
	args = parser.parse_args()

	main(args)