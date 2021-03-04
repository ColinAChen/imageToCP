import numpy as np
import cv2
import argparse
import math

def main(args):

	image = cv2.imread(args.pathToImage)
	#blank = np.zeros(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape)
	#blank[0:10, 0:10] = 255
	#showImage(blank, '00')
	
	# use hough line transform to determine the location of lines in the input image
	lines = parseImage(args.pathToImage)
	
	# convert the lines from parametric to cartesian coordinates and determine the bounds of each line
	finalLines = determineBounds(image, lines)
	
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
	image = cv2.imread(pathToImage)
	# convert to grayscale
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# threshold, send any color to black, all others to white
	ret, grayscale = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY_INV)
	# resize so that everythig is divisible by 2
	if (grayscale.shape[0] % 2 != 0):
		grayscale = cv2.resize(grayscale, (grayscale.shape[0] * 2,grayscale.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
	#grayscale = thresh
	#showImage(grayscale, 'grayscale')
	# thin lines so that hough transform doesn't pick up lines on both sides of thick lines
	

	#grayscale = thinLines(grayscale)
	

	#cv2.imwrite('gray.png', thresh)
	#thinned_name = pathToImage.split('.')[0] + '_thinned.png'
	#cv2.imwrite(thinned_name, grayscale)
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
	rows, cols, channels = image.shape
	lines = []
	print('\n')
	# keep track of a list of intersections, if a line is close enough to this line, snap it to the intersection by modifying
	# the starting and ending points

	# assume the corners of the square are intersections
	# maybe can also assume havles? The user might be able to specify this?
	intersections = set()
	intersections.add((0,0))
	intersections.add((0, rows-1))
	intersections.add((cols-1, 0))
	intersections.add((rows-1, cols-1))
	snapThreshold = rows/100 # might need to change later based on grid size
	#intersections = set((0,0), (0, rows-1), (cols-1, 0), (rows-1, cols-1))
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
			x1 = int(x0 + 1000*(-b))
			col1 = x1
			#x1 = max(0,x1)
			#x1 = min(cols, x1)

			y1 = int(y0 + 1000*(a))
			row1 = y1
			#y1 = max(0,y1)
			#y1 = min(rows, y1)

			x2 = int(x0 - 1000*(-b))
			col2 = x2
			#x2 = max(0,x2)
			#x2 = min(cols, x2)

			y2 = int(y0 - 1000*(a))
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
				startRow, startCol = start = getStart((row1,col1), slope, rows-1, vertical=vertical)
				offset = (startRow + slope, startCol + 1)
				snappedRow, snappedCol = snap(start, offset, intersections, snapThreshold)
				#snappedRow,snappedCol = startRow, startCol
				print('start',start)
				# iterate through once to check for intersections
				# for addCol in range(cols - int(startCol)):
				# 	# step through the line
				# 	# check if there are any existing intersections nearby
				# 	# if so, consider snapping to them to the intersection
				# 	# I think it is easier to modify the line in parametric than in coordinate
				# 	checkCol = startCol + addCol
				# 	checkRow = startRow + (slope * addCol)
				# 	snappedRow, 
				# 	# distanceToIntersections = [distance((checkRow, checkCol), intersection) for intersection in intersections]
				# 	# minDistance = min(distanceToIntersections)
				# 	# if (minDistance < snapThreshold):
				# 	# 	# find the intersection
				# 	# 	# assume the first one is fine
				# 	# 	minIndex = distanceToIntersections.index(minDistance)

				# 	# 	snappedRow, snappedCol = intersections[minIndex] # assume the intersections is part of the line

				# 	# 	# consider this point snapped, don't chceck others, might need to check for better matches later
				# 	# 	break
				# # iterate again to check for line bounds
				# # start at snapped coordinates and check both directions of slope to cover the whole line
				# # hopefully at this point, lines will intersect with other lines
				# # add new intersectiosn here

				# forward iteration
				startLineSegment = False
				startPoint = (-1,-1)
				endPoint = (-1,-1)
				for colOffset in range(cols - int(snappedCol)):
					checkRow = snappedRow + slope*colOffset
					checkCol = snappedCol + colOffset
					# round
					checkRowInt = int(round(checkRow))
					checkColInt = int(round(checkCol))
					# check if we are one a line
					if one line and startLineSegment = False:
						startLineSegment = True
						# snap if this is close enough to an intersection
						distanceToIntersections = [distance((checkRow, checkCol), intersection) for intersection in intersections]
						minDistance = min(distanceToIntersections)
						if (minDistance < snapThreshold):
						# find the intersection
						# assume the first one is fine
							minIndex = distanceToIntersections.index(minDistance)

						snappedRow, snappedCol = intersections[minIndex]
						startPoint = (checkCol, checkRow)
					elif offline and startLineSegment = True:
						startLineSegment = False
						# snap if this is close enough to an intersection
						# check for intersections and update the list of intersections
						line = (startPoint, endPoint)
						print(line)
						lines.append(line)


					neighbors = 
					image[]
				# backward iteration
				for colOffset in range(int(snappedCol), -1, -1):
					checkRow = snappedRow + slope*colOffset
					checkCol = snappedCol + colOffset
					# round
					checkRowInt = int(round(checkRow))
					checkColInt = int(round(checkCol))
					# check if we are one a line
					if one line and startLineSegment = False:
						startLineSegment = True
						startPoint = (checkCol, checkRow)
					elif offline and startLineSegment = True:
						startLineSegment = False
						line = (startPoint, endPoint)
						print(line)
						lines.append(line)


					


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

			line = ((x1,y1), (x2,y2))
			#print(line)
			#print('')

			#print(x0,y0,x1,y1,x2,y2)
			#print('\n')
			lines.append(line)
	print(lines)
	return lines

def snap(point1, point2, intersections, snapThreshold):
	# determine standard form of the line

	# calculate the distance between every intersection and this line
	distanceToIntersections = [distanceLineToPoint(point1, point2, intersection) for intersection in intersections]
	# get the shortest distance
	minDistance = min(distanceToIntersections)
	if (minDistance < snapThreshold):
		# if the minimum distance is withing snapThreshold away,
		# assume the intersections is part of the line and return the intersection
		minIndex = distanceToIntersections.index(minDistance)
		return intersections[minIndex]
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
	x2,p2 = p2
	x0,y0 = target
	((x2 - x1) * (y1 - y0)) * ((x1 - x0) * (y2 - y1))
	#numerator = abs((a * x) + (b * y) + c)
	numerator = ((x2 - x1) * (y1 - y0)) * ((x1 - x0) * (y2 - y1))
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

Arguments:
	line1 : ((x1, y1), (x2, y2)), line segment 1
	line2 : ((x1, y1), (x2, y2)), line segment 2
Returns:
	True if the line segments intersect, False otherwise

'''
def lineSegmentIntsersect(line1, line2):
	pass

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
	# #if (row == 0 or row == length):
	# elif (0 <= row <= length and 0 <= col <= length):
	# 	# point is inside the square
	# 	## point is on the bottom edge, there is a better start if the slope is negative
	# 	## negative slope means the line intersects the square on the left or top edge

		### GENERAL ALGORITHM ###
		# Reverse slope
		# check if changing column to zero sends the row to a point on the left edge
		# if not, the point must intersect on the top edge 
		# calculate where changing the row length times will send the column

		# if (slope > 0):
		# 	# positive slope will point us towards the right, therefore this point is optimal
		# 	return point

		# conceptually, -1 will either swap col in the case the col < 0 or swap slope inthe case that col > 0
		leftRow = (col * slope * -1)
		if (0 <= leftRow <= length):
			# intersection with the left edge is in bounds
			return (leftRow, 0)
		elif (leftRow < 0):
			# intersection with bottom edge
			return (0, col - (row/slope))
		else:
			# intersection with top edgess
			return (length, col - ((length-row) / slope))
		
	# # if (row == length):
	# # 	# point is on top edge, we can reach a better start if the slope is positive
	# # 	# positive slope means the line intersects the square on the left or bottom edge
	# # 	if (slope < 0):
	# # 		return point
	# # 	leftRow = length - (slope * (length - col))
	# # 	if (0 <= rightRow <= length):
	# # 		return (rightRow, 0)
	# # 	else:
	# # 		return (0, )
		

	# else:
	# 	# point is outside the square
	# 	# determine which direction 
	# 	if (col < 0):
	# 		# point is behind the left edge, simply send it along until it reaches the left edge
	# 		return (-1 * col * slope, 0)
	# 	elif (col > length):
	# 		# point is past the right edge
	# 		# find the itersection with the top, bottom, or left edge
	# 		# negative slope w


		
	


'''
Write a .cp file with a given list of lines

Arguments:
	pathToCP (str) : path to the cp file
	lines (list) : list of lines, output from determineBounds
Returns:
	No return but will create a file at path of pathToCP in the .cp file format
'''
def writeCP(pathToCP, lines):
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