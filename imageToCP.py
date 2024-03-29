from util import *
from lineTest import *

# TO DO
'''

On intersection, divide sements into two segments
Add slider for initial grayscale
Add input for minimum number of votes/min segment instead of trying to automate it

Add input for valley/mountain type
'''

'''
3/11/2023
use plant's idea of detecting verticies first to create a point set
Robert Lang's reference finder is designed with human level precision in mind
I wanted to get better precision than that, perhaps the new approach will lend better results
but this appraoch should be able to get something
We should be able to plug and play after that


Think about intersection as set vs list, I think I have to search no matter what but maybe set can save time if things happend to line up?



Notes:
	Resizing image in preprocess and cropSquare to 500,500 for now
	Stop creating the thinned images, I want to see if we can just average 

'''
def main(args):

	#print(args)
	#print(args.pathToImage)
	image = cv2.imread(args.pathToImage, cv2.IMREAD_UNCHANGED)
	image = cropSquare(image)
	showImage(image,'square')
	referenceImage = image
	if(args.pathToImage.split('.')[1] == 'png'):
		# replace transparent background of png with white
		# this is important for line detection
		#https://stackoverflow.com/questions/53732747/set-white-background-for-a-png-instead-of-transparency-with-opencv/53737420
		#make mask of where the transparent bits are
		trans_mask = image[:,:,3] == 0
		#replace areas of transparency with white and not transparent
		image[trans_mask] = [255, 255, 255, 255]
		#new image without alpha channel...
		referenceImage = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	showImage(referenceImage,'ref')
	#showImage(referenceImage,'referenceImage')
	#testColor = referenceImage.copy()

	#testColor[0:100,0:100] = [0,0,255]
	#BLUE = [255,0,0]
	#RED = [0,0,255]
	#showImage(testColor, 'testColor')
	#image = cv2.imread(args.pathToImage)
	#blank = np.zeros(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape)
	#blank[0:10, 0:10] = 255
	#showImage(blank, '00')
	
	'''
	3/12/2023 11:21
	TODO: remove options for line thinning, it is definitely bait
	'''

	# use hough line transform to determine the location of lines in the input image
	if (args.pathToThinned):
		# check if the there is a file at pathToThinned
		if os.path.isfile(args.pathToThinned):
			# there is a file at pathToThinned, assume it is legit and use it to calculate lines
			thinnedImage = cv2.imread(args.pathToThinned)
			#lines, pointSet = parseImage(thinnedImage,pathToImage=args.pathToImage, pathToThinned=args.pathToThinned,stepAngle=args.stepAngle, grid=args.grid)

			lines, pointSet = parseImage(referenceImage,pathToImage=args.pathToImage, pathToThinned=args.pathToThinned,stepAngle=args.stepAngle, grid=args.grid)
		else:
			# no file was found so we must create one
			lines, pointSet = parseImage(referenceImage, pathToThinned=args.pathToThinned, needThinLines=True, stepAngle=args.stepAngle, grid=args.grid)
	else:
		# no path to thinned was specified, we will run the line thinning algorithm and save the image at an automatically generated path
		# might change later to not save at all
		lines, pointSet = parseImage(referenceImage, pathToImage=args.pathToImage,needThinLines=True, stepAngle=args.stepAngle, grid=args.grid)
	
	# convert the lines from parametric to cartesian coordinates and determine the bounds of each line
	#finalLines = determineBounds(expand(deblur(referenceImage)), lines, grid=args.grid, startingAngle=args.angle, pointSet=pointSet)
	referenceImage = addBuffer(referenceImage)
	colorThreshold(referenceImage)
	finalLines = determineBounds(referenceImage, lines, grid=args.grid, startingAngle=args.angle, pointSet=pointSet)

	# write these lines to a cp file
	cpFileName = args.pathToImage.split('.')[0] + '.cp'
	writeCP(cpFileName, finalLines, show=True)
	

'''
Read an image, perform preprocessing, and return a dictionary of angles:distances from origin

Arguments:
	pathToImage (str) : path to image to read and convert to .cp
Returns:
	degreesToRho {degrees (float) : [(distance from origin (float), radians from origin (float)) , ...]}
		For a degree we find a line, retrieve a list of distances and angles in radians. 
'''
def parseImage(image, pathToImage='',pathToThinned='',needThinLines=False, stepAngle=0, grid=0):

	# add a buffer around the edges include the edges in the intersection finding process. This way we won't need to add any points manually
	
	#showImage(image)
	buf = 10
	image = addBuffer(image)
	showImage(image, 'received from main')
	if (needThinLines):
		image = cv2.resize(image, (500,500))
		image = addBuffer(image)
		showImage(image, title='resize')
		

		print('no thinned lines image provided, running line thinning algorithm')
		grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ret, grayscale = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY)
		# thin lines so that hough transform doesn't pick up lines on both sides of thick lines
		showImage(grayscale, 'image before thinning')
		grayscale = thinLines(grayscale)
		#grayscale = zhangSuen(grayscale)

		#cv2.imwrite('gray.png', thresh)
		if (pathToThinned != ''):
			thinned_name = pathToThinned
		else:
			thinned_name = pathToImage.split('.')[0] + '_thinned.png'
		cv2.imwrite(thinned_name, grayscale)
		#showImage(image, 'image')
		#showImage(thinned, 'thinned')
		#grayscale = thinned
	
		#grayscale = cv2.imread(pathToThinned,cv2.IMREAD_GRAYSCALE)
	else:
		print('thinned received')
		#grayscale = image
		grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#showImage(grayscale)
	showImage(grayscale, 'before threshold', scale=True)
	
	ret, grayscale = cv2.threshold(grayscale, 150, 255, cv2.THRESH_BINARY_INV)
	#ret, grayscale = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)
	showImage(grayscale, 'thinned', scale=True)
	


	#grayscale = cv2.imread('fish_thinned.png', cv2.IMREAD_GRAYSCALE)
	#showImage(grayscale, 'fish_thinned')
	# perform line detection
	edges = cv2.Canny(grayscale, 50, 150, apertureSize=3)

	showImage(edges, 'edges')
	#print(edges.shape)
	showImage(edges, 'edges', scale=True)
	#print(edges)
	# maybe change votes to be based on the square size/grid size
	# I can see this being a potential issue 
	
	
	print('using grid of ',grid)
	
	#votes = 20
	# reshape if we are using the thinned image as input to the hough transform 
	# instead of the edge detector
	rows,cols = (-1,-1)
	if (len(grayscale.shape) == 2):
		rows,cols = grayscale.shape
	else:
		rows,cols,channels = grayscale.shape
	#print(grayscale.shape)
	#votes = int(rows/20)
	#if (grid != 0):
		#votes = int(min(rows,cols) / grid) - 1
	votes = 40
	print('minimum votes required: ', votes)
	angle = (stepAngle/360) * 2 * np.pi
	print('using angle step size of ',stepAngle, ' degrees, ',angle,' radians')
	lines = cv2.HoughLines(edges,.5,angle,votes)
	#lines = cv2.HoughLines(grayscale, .5, angle, votes)
	# try probabilistic hough transform to estimate endpoints
	blank = np.zeros(grayscale.shape)
	#grayscale = cv2.imread(pathToImage)
	#grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)
	#showImage(grayscale,'test')
	# lines = cv2.HoughLinesP(grayscale, .5, angle, votes)
	# if lines is not None:
	# 	for i in range(0, len(lines)):
	# 		l = lines[i][0]
	# 		cv2.line(blank, (l[0], l[1]), (l[2], l[3]), (255,255,255), 2, cv2.LINE_AA)
	# showImage(blank, 'houghlinesP_thinned')

	# blank = np.zeros(grayscale.shape)
	# lines = cv2.HoughLinesP(edges, .5, angle, votes)
	# if lines is not None:
	# 	for i in range(0, len(lines)):
	# 		l = lines[i][0]
	# 		cv2.line(blank, (l[0], l[1]), (l[2], l[3]), (255,255,255), 2, cv2.LINE_AA)
	# showImage(blank, 'houghlinesP_edges')

	
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
				'''
				3/12/2023 11:20am
				Averaging on an extended view seems to work very well, line thinning was bait
				'''
				addToList = True
				for i,parametricLine in enumerate(degreesToRho[degrees]):
					#print(parametricLine)
					if (abs(parametricLine[0] - rho) < 10):

						print('averaging lines at angle', degrees)
						#if (degrees == 90 or degrees == 0):
						#	#print('update rho from ', parametricLine[0], ' to ', ((parametricLine[0] + rho)/2))
						#	#print(degreesToRho[degrees][i])
						degreesToRho[degrees][i] = (((parametricLine[0] + rho)/2), parametricLine[1])
						#if degrees == 90 or degrees == 0:
						#	#print(degreesToRho[degrees][i])
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
			#if (random.randint(0,20) < 5):
			cv2.line(blank,(x1,y1),(x2,y2),(255,255,255),2)
			#showImage(blank, 'line')
			#blank = np.zeros(blank.shape)
	#cv2.line(blank, (0,0), blank.shape, (255,0,0),2)
	#cv2.line(blank, (0,blank.shape[0]), (blank.shape[0],0), (255,0,0),2)
	#print(blank.shape)
	#showImage(blank, 'blank w/ lines', scale=True)
	blank = np.zeros(image.shape)
	temp = np.zeros(image.shape)
	showLines = np.zeros(image.shape)
	#print(blank.shape)
	for degrees in degreesToRho:
		print(degrees)
		print(degreesToRho[degrees])
		#if (degrees != 90):
		for rho in degreesToRho[degrees]:
			angle = degrees * 2 * np.pi / 360
			a = np.cos(angle)
			b = np.sin(angle)
			x0 = a*rho[0]
			y0 = b*rho[0]
			x1 = int(x0 + 2000*(-b))
			y1 = int(y0 + 2000*(a))
			x2 = int(x0 - 2000*(-b))
			y2 = int(y0 - 2000*(a))
			#print(x0,y0,x1,y1,x2,y2)
			#print('\n')
			#if (random.randint(0,20) < 5):
			cv2.line(temp,(x1,y1),(x2,y2),(1,1,1),2)
			blank += temp
			temp = np.zeros(blank.shape)
			cv2.line(showLines,(x1,y1),(x2,y2),(255,255,255),2)
	showLines = showLines[buf:buf+500, buf:buf+500, :]
	blank = blank[buf:buf+500, buf:buf+500, :]
	showImage(showLines, title='hough lines')
	gs = blank[:,:,0]
	gs = np.where(gs>1, 255,0)
	gs = np.uint8(gs)
	blank[:,:,0] = gs
	blank[:,:,1] = gs
	blank[:,:,2] = gs
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(gs)
	
	pointSet = set()
	for c in centroids:
		x,y = c

		# back refernce the image to see if there are any lines here
		# actually let's just push on for now, I think plant's method is better
		pointSet.add((int(y)+buf, int(x)+buf))


	#print(labels)
	#print(stats)
	#print(centroids)
	showImage(blank, 'blank average', scale=True)
	blank = np.zeros(image.shape)
	blank[:,:,:] = image
	for p in pointSet:
		col, row = p
		#sc = col-buf
		#sr = row-buf
		# if row < 0 or row > rows-1 or col < 0 or col > cols-1:
		# 	continue
		#row += buf
		#col += buf
		blank[row-2:row+2,col-2:col+2,:] = (0,255,0)
		#blank[row][col] = (255,255,255)
	showImage(blank)
	cv2.imwrite('cc.png', blank)

	#print(degreesToRho)
	return degreesToRho, pointSet
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
Reconstruct the crease pattern by running along the lines we found and referencing the original image

Arguments:
	image (image as numpy array) : original image of crease pattern, ideally in color to determine line type
	degreesToRho (dict) : dictionary mapping degreees to a list of (rho, theta in radians) values, output from parseImage
Returns:
	lines (list of tuples) : [((x1,y1), (x2,y2)), ....] list of lines represented as tuples. Tuples are (start point, end point)
'''
def determineBounds(image, degreesToRho, grid=0, startingAngle=0, pointSet=set()):
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
	
	#snapThreshold = 100 #rows/100 # might need to change later based on grid size
	#snapThreshold = rows/20
	snapThreshold = 10
	print('using snap threshold of ', snapThreshold)
	#if (grid != 0):
	#	snapThreshold = rows/(2*grid)
	#corners = [(0,0), (0, rows-1), (cols-1, 0), (rows-1, cols-1)]
	#midpoints = [(0,(cols-1)/2), ((rows-1)/2, 0), ((rows-1)/2, cols-1), (rows-1,(cols-1)/2)] # can add grid lines here later
	#intersections = [(0,0), (0, rows-1), (cols-1, 0), (rows-1, cols-1)] # consider adding grid lines as bounds here?
	#intersections = corners# + midpoints
	intersection = []
	if (grid > 1):
		# grid = 1 should just return corners
		# grid = 2 is bisection
		print('creating snaps points for grid', grid)
		gridSize = grid
		#gridSize = 4
		for i in range(gridSize+1):
			for j in range(gridSize+1):
				if ((i * (rows-1)/gridSize, j * (cols-1)/gridSize) not in intersections):
					intersections.append((i * (rows-1)/gridSize, j * (cols-1)/gridSize))
	intersections = (list(pointSet))
	# skip this for now, we will use the point set from above
	'''
	if (startingAngle == 22.5):
		print('creating snap points for starting angle', startingAngle)
		# top edge, bottom edge, left edge, right edge, bottom left, bottom, right, top left, top right
		# add edge intersections with 22.5 folds
		# bottom edge
		first = (cols-1) * math.tan(math.pi/8)
		second = (cols-1) * (1 - math.tan(math.pi/8))
		intersections.append((0, first))
		intersections.append((0, second))
		# top edge
		intersections.append((rows-1, first))
		intersections.append((rows-1, second))
		# left edge
		intersections.append((first, 0))
		intersections.append((second, 0))
		# right edge
		intersections.append((first, cols-1))
		intersections.append((second, cols-1))
		# add intersections betwen 22.5 and diagonals
		left = (rows-1) / (1 + (math.tan(3 * (math.pi/8))))
		right = (rows-1) / (1 + math.tan(math.pi/8))
		#print('left', left)
		#print('right', right)
		intersections.append((left, left))
		intersections.append((left,right))
		intersections.append((right,left))
		intersections.append((right,right))
		# add intersections betwen 22.5 and 22.5
		intersections.append(((rows-1)/2, first/2))
		intersections.append(((rows-1)/2, second + (first/2)))
		intersections.append((first/2, (cols-1)/2))
		intersections.append((second + (first/2), (cols-1)/2))
	#print(intersections)
	'''

	#toolbar_width = 15
	# setup toolbar
	#sys.stdout.write("[%s]" % (" " * toolbar_width))
	#sys.stdout.flush()
	#sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

	for p in intersections:
		print(p)
	completedAngles = 0

	#sys.stdout.write("completed angles: %d/%d" % (completedAngles, len(degreesToRho)))
	#sys.stdout.flush()

	#sys.stdout.write("\b" * (18))



	# this will create duplicated segments if any points end on the edges, need to revist this
	# find all the lines that touch the edges/corners
	# add these lines first so that as we add lines
	# we can reconstruct them as we find intersections

	'''
	# add the corners to lines to create a square
	# top edge
	lines[(corners[0], corners[1])] = 1
	# left edge
	lines[(corners[0], corners[2])] = 1
	# right edge
	lines[(corners[2], corners[3])] = 1
	# bottom edge
	lines[(corners[1], corners[3])] = 1
	'''

	for angle in degreesToRho:
		print('working on lines with angle ',angle)
		for parametricLine in degreesToRho[angle]:
			# convert parametric coordinates to cartisian to prepare for referencing against the image
			a = np.cos(parametricLine[1])
			b = np.sin(parametricLine[1])
			# x0 = a*degreesToRho[angle][0]
			# y0 = b*degreesToRho[angle][0]
			'''
			3/11/2023 10:52pm
			We need to account for the offset from the lines image
			hardcode for now, maybe try to fix this upstream later

			nvm lets do everything in the buffered image so we naturally get start and end
			'''
			x0 = a*(parametricLine[0]-10)
			y0 = b*(parametricLine[0]-10)
			x0 = a*(parametricLine[0])
			y0 = b*(parametricLine[0])
			#print('x0',x0)
			#print('y0',y0)
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
			#print(x1,y1,x2,y2)
			vertical = False
			start1 = True
			slope = 0
			if (col2-col1 == 0):
				# no change in col == no change in x
				vertical = True
				#print('slope: vertical')
			elif (row2 - row1) > 0:
				# x1 towards x2 is positive
				slope = (row2 - row1) / (col2 - col1)
			else:
				# x2 towards x1 is positive
				# get the slope with increasing column/y
				slope = (row1 - row2) / (col1 - col2) 
				start1 = False
			#print('send slope:',slope)
			# walk along the line to determine the real bounds
			# in this step, we will reference the original and try to rebuild the crease pattern with intersection accuracy
			# first pass will check if this line is reasonably close to an intersection, if so, we can snap the line to it
			# second pass will reference the original line to determine line bounds and and crease types if the image is in color
			
			# compeletely arbitrary
			# TO DO determine min semgment length when grid is 0
			# this should probably be the same as the number of votes?
			#minSegmentLength = rows/20
			minSegmentLength = 10
			if (grid != 0):
				minSegmentLength = int(min(rows,cols) / grid) - 1
			minSegmentLength = 10
			if not vertical:
				# # forward pass
				# referenceImage((row1, col1), slope, intersections, lines, image, snapThreshold, forward=True, vertical=False, minSegmentLength=minSegmentLength)
				# # backward pass
				# referenceImage((row1, col1), slope, intersections, lines, image, snapThreshold, forward=False, vertical=False, minSegmentLength=minSegmentLength)
				# forward pass
				simpleReferenceImage((row1, col1), slope, intersections, lines, image, snapThreshold, forward=True, vertical=False, minSegmentLength=minSegmentLength)
				# backward pass
				simpleReferenceImage((row1, col1), slope, intersections, lines, image, snapThreshold, forward=False, vertical=False, minSegmentLength=minSegmentLength)
			
			else:
				# vertical line, slope doesn't matter
				#referenceImage((row1, col1), -1, intersections, lines, image, snapThreshold, forward=True, vertical=True, minSegmentLength=minSegmentLength)
				simpleReferenceImage((row1, col1), -1, intersections, lines, image, snapThreshold, forward=True, vertical=True, minSegmentLength=minSegmentLength)
	#print(lines)
	#print('intersections',intersections)


	#sys.stdout.write("]\n") # this ends the progress bar
	return lines
'''
Code is repeated three times
1 forward iteration from start point
2 backward iteration from start point
3 vertical line edge case to handle no change in column
'''
def referenceImage(start, slope, intersections, lines, image, snapThreshold, forward=True, vertical=False, grid=0, minSegmentLength=10):
	'''
	In this step, we take the lines we calculated in parseImage, and attempt to create a crease pattern by referecing the original image.
	The lines we receive in parseImage have no endpoints, so we must figure out the endpoints in this step.
	The fact that this is an image means that the pixels values we observe can only approximate the actual points these crease make.
	We will recreate the crease pattern by using some knowledge of the crease pattern (for example, a grid size, starting angle, etc). 

	To recreate the lines, iterate over all the pixels this line passes through on the square
	We determine the bounds with a simple state machine

	'''
	# grab from util.py
	#RED = (0,0,255)# Red = (0, 0, 255)
	#BLUE = (255,0,0)# Blue = (255, 0, 0)
	
	colorThreshold = 60 # totally arbitary, not sure how this will play out for shorter line segments
	rows,cols,channels = image.shape
	
	row1,col1 = start
	blank = np.zeros(image.shape)
	#print('slope', slope)
	startRow, startCol = start = getStart((row1,col1), slope, rows-1, vertical=vertical)

	offset = (startRow + slope, startCol + 1)
	if (vertical):
		# no change in col as row changes
		offset = (startRow + 1, startCol)
	snappedStart, snappedStartOffset = snap(start, offset, intersections, snapThreshold, start=True)
	snappedRow, snappedCol = snappedStart

	#print('start iteration',(snappedRow, snappedCol))

	# loop is the same
	# define loop bounds
	bounds = range(1)
	if (vertical):
		# vertical line
		bounds = range(rows)
		#print ('vertical bounds', bounds)
	elif(forward):
		# forward iteration
		bounds = range(cols-int(snappedCol))
		#print('forward pass bounds', bounds)
	else:
		# backward iteration
		bounds = range(0,-int(snappedCol), -1)
		#print('backward pass bounds', bounds)
	
	# variables for line recreation state machine
	startLineSegment = False
	startPoint = (-1,-1)
	endPoint = (-1,-1)
	# 
	rollingAverageColor = [0,0,0] #currently getting rounded, not sure how that is affecting the accuracy
	startColor = [0,0,0]
	# consider changing to most populous color
	# dict {color : frequency}
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
		# if not forward and not vertical:
		# 	print(offset,(checkRow,checkCol))
		if (vertical):
			# handle the vertical line caes seperately
			checkRow = offset
			checkCol = snappedCol
		# keep track of the last point we visit in case we have an unfinished line
		finalRow, finalCol = (checkRow, checkCol)
		# round to access pixels in the original image
		# one problem could be that lines are too thin and rounding takes it off a line?
		checkRowInt = int(round(checkRow))
		checkColInt = int(round(checkCol))
		if (checkRowInt < 0 or checkRowInt >= rows or checkColInt < 0 or checkColInt >= cols):
			#finalRow, finalCol = (checkRow, checkCol)
			# protect image from referencing points out of bounds
			#print('out of bounds, leaving iteration')
			continue
		# check if we are one a line
		# might need to check neighbors, depending on how the row and column get rounded
		# if intensity != white, we are on a line
		# in the future, check the color
		# if (checkRowInt == 500):
		# 	print('row, col, intesity', checkRowInt, checkColInt, image[checkRowInt][checkColInt])

		# a line segment ends if it is different than the previous color
		# keep track of the starting color, end the semgment if the color is different
		# it's possible we won't be able to continue occluded lines?
		# maybe try to combine lines after the fact?
		if not colorEquals(image[checkRowInt][checkColInt], (255,255,255)) and not startLineSegment:
			#print('start line at',checkRow, ',',checkCol,'with intensity', image[checkRowInt][checkColInt])
			startLineSegment = True
			# snap if this is close enough to an intersection
			offset = (checkRow + slope, checkCol + 1)
			startPoint, startOffset = snap((checkRow, checkCol), offset, intersections, snapThreshold)
			
			# keep track of the starting color
			# assign start color as max of red or blue for checking later
			startColor = image[checkRowInt][checkColInt]
			print('start segment at', startPoint, ' with color ', startColor)
			# keep track of the color as a rolling average
			# this will allow us to determine the color of lines even if they are occluded by differently colored lines
			#rollingAverageColor = image[checkRowInt][checkColInt]
			#linePixels = 1



			highlightPoint((checkColInt,checkRowInt),image.copy())

		# don't need to do anything along a line
		elif not colorEquals(image[checkRowInt][checkColInt], startColor) and startLineSegment:
			# we have started a line and just encountered a white pixel
			# or another color line (could be a line intersection)
			# this means the line we are on has finished
			startLineSegment = False

			print('ending line at ', checkRowInt, checkColInt, ' with color ', image[checkRowInt][checkColInt])
			# snap if this is close enough to an intersection
			# check for intersections and update the list of intersections
			endPoint, endOffset = snap((checkRow, checkCol), startPoint,intersections, snapThreshold)
			line = (startPoint, endPoint)
			print('determined line ', line)
			if (distance(line[0], line[1]) > minSegmentLength):
				# determine the line type
				# opencv pixel intensities are returned as BlueGreenRed
				# print('rollingAverageColor for line:',rollingAverageColor)
				# lineType = 1
				# if (distance(rollingAverageColor, RED) < colorThreshold):
				# 	# red line indicates mountain fold
				# 	# .cp -> mountain fold is 2
				# 	print('mountain fold')
				# 	lineType = 2
				# elif (distance(rollingAverageColor, BLUE) < colorThreshold):
				# 	# blue line indiciates valley fold
				# 	# .cp -> valley fold is 3
				# 	print('valley fold')
				# 	lineType = 3
				lineType = getLineType(startColor)
				# add all new intersections this line creates to the list of intersections
				#addToIntersections(line, lines, intersections)
				addToIntersections(line, lines, intersections, lineType)
				

				print('end line', line)
				showLine(line, blank.copy(), lineType=lineType)
				

				#blank = np.zeros(blank.shape)
				#print(line)
				#lines.append(line)
				lines[line] = lineType
			else:
				print('line found is too short: ', distance(line[0], line[1]))
				#print('startLineSegment: ', startLineSegment)
				pass
				#print('line found is too short')

	if (startLineSegment):
		#print('ending line out of bounds ', checkRowInt, checkColInt, ' with color ', image[checkRowInt][checkColInt])
		startLineSegment = False
		#print('need to finish line segment')
		# finish unfinished line segments
		endPoint, endOffset = snap((finalRow, finalCol), startPoint,intersections, snapThreshold)
		line = (startPoint, endPoint)
		# endpoint might be off the square

		if (0<=line[1][0]<=rows-1 and 0<=line[1][1]<=rows-1 and distance(line[0], line[1]) > minSegmentLength):
			# determine the line type
			# opencv pixel intensities are returned as BlueGreenRed
			#print('rollingAverageColor for line:',rollingAverageColor)
			# lineType = 1
			# if (distance(rollingAverageColor, RED) < colorThreshold):
			# 	# red line indicates mountain fold
			# 	# .cp -> mountain fold is 2
			# 	print('mountain fold')
			# 	lineType = 2
			# elif (distance(rollingAverageColor, BLUE) < colorThreshold):
			# 	# blue line indiciates valley fold
			# 	# .cp -> valley fold is 3
			# 	print('valley fold')
			# 	lineType = 3
			lineType = getLineType(startColor)
			# add all new intersections this line creates to the list of intersections
			addToIntersections(line, lines, intersections, lineType)
			

			print('end line', line)
			showLine(line, blank.copy(), lineType=getLineType)
			

			#blank = np.zeros(blank.shape)
			#print(line)
			#lines.append(line)
			lines[line] = lineType
		else:
			pass
			#print('line found is too short')




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
def snap(point, offset, intersections, snapThreshold, start=False):
	print('\nfind snap point for ', point, ' using snapThreshold ', snapThreshold)
	### POSSIBLE IMPROVEMENTS?
	# Snap towards edges
	# Snap towards lines that don't already intersect?
	# Utilize returned offset to snap both ends? not sure how much this matters since fish base collapsed okay
	# start point should have different rules, I only care about points close to the line, I don't care about distnace to acutal points
	# that is the whole point of a forward and backward iteration
	row,col = point
	### also return the snap offset so we can offset the whole line?
	#print('find snap for point: ', point)
	#print('using offset:', offset)
	# calculate the distance between every intersection and this line
	distanceToIntersections = [round(distanceLineToPoint(point, offset, intersection),4) for intersection in intersections]
	if (start):
		# if this is a starting point, we only care about the closest point to the line
		#print('finding starting point, only check for proximity to line')
		#print(distanceToIntersections)
		minDistance = min(distanceToIntersections)
		# assume that ties mean the points are on the same line
		if (minDistance < snapThreshold):
			minIndex = distanceToIntersections.index(minDistance)
			snapPoint = snapRow, snapCol = intersections[minIndex]
			#print('min distance', minDistance, ' to point', snapPoint)
			return snapPoint, (snapRow-row, snapCol-col)
		else:
			return point, (0,0)
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
	print('minDistance', minDistance)
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
		print('no points near line, snap to', point)
		return point, (0,0)

'''
Find the closest point in the point set
This assumes that all intersections have been found ahead of time
'''
def simpleSnap(point, intersections, threshold=50):
	distanceToIntersections = [distance(point,x) for x in intersections]
	#print(distanceToIntersections)
	minDistance = min(distanceToIntersections)
	#print(minDistance)
	if minDistance > threshold:
		print("can't find snap point for ",point, 'with threshold',threshold)
		for p,d in zip(intersections,distanceToIntersections):
			print(p,':',d)
		#print(distanceToIntersections)
		return None
	minIndex = distanceToIntersections.index(minDistance)
	#print(minIndex)
	return intersections[minIndex]

'''
Given a slope, find the start
Travel until we find a line in the reference image
If there is no snap point, we can probably ignore
Snap to the closest snap point
Travel along line until it stops
Snap to the closest snap point

Repeat until we reach the end of the line
'''
def simpleReferenceImage(start, slope, intersections, lines, image, snapThreshold, forward=True, vertical=False, grid=0, minSegmentLength=10):

	# add a buffer to the image
	buf = 10
	
	#image = addBuffer(image)
	rows,cols,channels = image.shape
	#print(buff.shape)
	#image = np.uint8(image)
	#colorThreshold = 300 # totally arbitary, not sure how this will play out for shorter line segments
	
	
	row1,col1 = start
	#print('slope', slope)
	#colorThreshold(image)

	print("intial start:", start)

	# find the intersection with the leftmost point
	startRow, startCol = start = getStart((row1,col1), slope, rows-1, vertical=vertical)
	#highlightPoint((startRow, startCol),image, supress=False)
	print('before',startRow, startCol)
	#startRow, startCol = simpleSnap((startRow, startCol), intersections)
	#print('snap start:', (startRow, startCol))
	startRow = int(startRow)# + buf
	startCol = int(startCol)# + buf
	#highlightPoint((startRow, startCol),image, supress=False)
	#highlightPoint((startRow, startCol),buff, supress=False)
	# start inside the square?
	# use this to show which direction we are traveling?
	# offset = (startRow + slope, startCol + 1)
	# if (vertical):
	# 	# no change in col as row changes
	# 	offset = (startRow + 1, startCol)
	'''
	start at startRow and startCol
	travel until we get to a line (decide by color), maybe sliding window later?

	when we hit a line, simple snap to the closest point
	continue traveling along the original path which should be more accurate
	when we hit the end of the line, snap to the closest point and add that line to the set
	repeat

	'''
	first = None
	#currentColor = [0,0,0]
	kernel = 7


	bounds = range(1)
	if (vertical):
		# vertical line
		bounds = range(rows)
		#print ('vertical bounds', bounds)
	elif(forward):
		# forward iteration
		bounds = range(cols-int(startCol))
		#print('forward pass bounds', bounds)
	else:
		# backward iteration
		bounds = range(0,-int(startCol), -1)
		#print('backward pass bounds', bounds)

	print('slope:',slope)
	print('start:',start)
	print('bounds:',bounds)

	sr = min(rows-1-kernel, startRow)
	sc = min(cols-1-kernel, startCol)
	#print(image[sr:sr+kernel,sc:sc+kernel])
	#print(image[sr:sr+kernel,sc:sc+kernel].flatten())
	#startSet = set(image[sr:sr+kernel,sc:sc+kernel].flatten())
	rs = max(0, startRow - kernel)
	re = min(rows-1,startRow + kernel)
	cs = max(0, startCol - kernel)
	ce = min(cols-1,startCol + kernel)
	
	print(rs, re, cs, ce)
	#startSet = getColorSet(image[rs:re, cs:ce])
	startSet = colorOrder(image[rs:re, cs:ce])

	print('start set:', startSet)
	startSet = ()
	showImage(image[rs:re, cs:ce])

	

	for offset in bounds:
		checkRow = startRow + (slope*offset)
		checkCol = startCol + offset
		# if not forward and not vertical:
		# 	print(offset,(checkRow,checkCol))
		if (vertical):
			# handle the vertical line caes seperately
			checkRow = startRow + offset
			checkCol = startCol	

		#finalRow, finalCol = (checkRow, checkCol)
		checkRowInt = int(round(checkRow))
		checkColInt = int(round(checkCol))
		#print('check:',(checkRowInt, checkColInt))
		if (checkRowInt < 0 or checkRowInt >= rows or checkColInt < 0 or checkColInt >= cols):
			#finalRow, finalCol = (checkRow, checkCol)
			# protect image from referencing points out of bounds
			#print('out of bounds, leaving iteration')
			continue # not break because we might out of bounds

		# color has changed, current line has changed
		# consider adding a buffer to the reference image and doing a sliding window, this could help detect different thickness lines


		#colorDist = distance(currentColor, image[checkRowInt][checkColInt])
		rs = max(0, checkRowInt - kernel)
		re = min(rows-1,checkRowInt + kernel)
		cs = max(0, checkColInt - kernel)
		ce = min(cols-1,checkColInt + kernel)
		
		currentSet = colorOrder(image[rs:re, cs:ce])#set(image[sr:sr+kernel,sc:sc+kernel,:].flatten())
		#print('current color order: ',currentSet)
		#showImage(image[rs:re, cs:ce])
		#if colorDist > colorThreshold:
		if startSet != currentSet:
			print('color changed from ', startSet,'to',currentSet, 'at',(checkRowInt, checkColInt))
			'''
			# don't need to do anything if the same color is in the lead?
			'''
			showImage(image[rs:re, cs:ce])
			# line has changed
			if first is None:

				# this is the first point in the line
				first = simpleSnap((checkRow, checkCol), intersections)
				print('start line segment at',first)

			else:
				# we have already found a point in the line
				# for now assume that there will be a close snap point
				# I'm not sure if this will always be true
				second = simpleSnap((checkRow, checkCol), intersections,threshold=9999)
				print('finish line segment at',second)

				# check if this line is long enough (use this to reject passing through points)
				if distance(first, second) < minSegmentLength or startSet == ():
					print("throw away segment of length", distance(first, second))
					print('throw away segment with color order: ', startSet)
					# this line is too short, or was white, throw away the current line segment
					first = simpleSnap((checkRow, checkCol), intersections, threshold=9999)
					print("start new segment at ", first)
				else:
					
					lineType = getLineType(startSet[0])
					#lineType = getLineType(currentColor)
					lines[(first,second)] = lineType
					print("add line",(first,second),lineType)
					first = second
					print('start new segment at',first)
			#highlightPoint((checkRowInt, checkColInt),image, supress=False)
			#currentColor = image[checkRowInt][checkColInt]
			startSet = currentSet
			print("")

	# if first is not None:
	# 	# finish an incomplete line
	# 	# might be able to get around this with buffer and border
	# 	print("didn't finish line")



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
	#print('finding start for', point)
	if (vertical):
		# this is a vertical line, return the column intersection with row 0
		return (10, col)
	elif (slope == 0):
		# this is a horizontal line, return the row intersection with column 0
		return (row,10)
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
		#print('left row', leftRow)
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
Crop the square crease pattern out of an image

Arguments:
	image (image as numpy array) : image to crop a square creaes pattern out of
Returns:
	square (image as numpy array) : square image of numpy array
'''
def cropSquare(image):
	rows,cols, channels = image.shape
	# find top left corner
	left = -1
	top = -1
	for row in range(rows):
		for col in range(cols):
			if sum(image[row][col]) < 220*3:
			#if not colorEquals((255,255,255), image[row][col]):
				top = row
				left = col
				break
	right = -1
	bottom = -1
	for row in range(rows-1,-1,-1):
		for col in range(cols-1,-1,-1):
			#if not colorEquals((255,255,255), image[row][col]):
			if sum(image[row][col]) < 220*3:
				bottom = row
				right = col
				break
	print('length', (right-left))
	print('height', (top-bottom))
	bounds = image[bottom:top, left:right]
	bounds = cv2.resize(bounds, (500,500))
	return bounds#image[bottom:top, left:right]

def addBuffer(image, buf=10):
	rows, cols, channels = image.shape
	color = BLACK
	backgroundColor = WHITE
	if np.mean(image) < 127:
		# background is black

		color = WHITE
		backgroundColor = BLACK
	temp = np.zeros(image.shape)
	temp[:,:,:] = image
	temp[:,0,:] = color
	temp[:,cols-1,:] = color
	temp[0,:,:] = color
	temp[rows-1,:,:] = color

	
	out = np.zeros((rows+(2*buf), cols+(2*buf), 3))
	out[:,:,:] = backgroundColor
	out[buf:rows+buf, buf:cols+buf,:] = temp
	#showImage(out)
	temp = out
	temp = np.uint8(temp)

	return temp
def colorThreshold(image):
	# convert image to only have blue, red, black, white
	rows, cols, c = image.shape
	for r in range(rows):
		for c in range(cols):
			color = image[r][c]
			colors = [WHITE, BLACK, BLUE, RED]
			d = [distance(color, c) for c in colors]
			setColor = colors[d.index(min(d))]
			image[r][c] = setColor
	showImage(image, title='color threshold')
def getColorSet(patch):
	ba = patch[:,:,0].flatten()
	ga = patch[:,:,1].flatten()
	ra = patch[:,:,2].flatten()
	colors = [(b,g,r) for b,g,r in zip(ba,ga,ra)]
	return set(colors)

'''
Crappy hack for now
'''
def colorFromColorSet(colorSet):
	removeOrder = [WHITE, BLACK]
	if WHITE in colorSet:
		colorSet.remove(WHITE)
	if (len(colorSet)) == 1:
		return list(colorSet)[0]
	if BLACK in colorSet:
		colorSet.remove(BLACK)
	if len(colorSet) == 1:
		return list(colorSet)[0]
	else:
		return BLACK

'''
When we are following a line, assume the most common color besides white will be the correct color
In a window, get the number of black, blue, and red
# only include colors that are actually present

# In need this to return the same color order in the event of ties
# ignore white, we don't actually care about it
'''
def colorOrder(patch):
	rows, cols, c = patch.shape
	b = patch[:,:,0]
	g = patch[:,:,1]
	r = patch[:,:,2]

	numWhite = np.sum(np.where(g == 255, 1, 0))
	numRed = np.sum(np.where(r == 255, 1, 0)) - numWhite
	numBlue = np.sum(np.where(b == 255, 1, 0)) - numWhite
	numBlack = (rows * cols) - (numWhite+numBlue+numRed)


	#numWhite = np.sum(np.where(patch == WHITE, 1, 0))
	#numBlack = np.sum(np.where(patch == BLACK, 1, 0))
	#numBlue = np.sum(np.where(patch == BLUE, 1, 0))
	#numRed = np.sum(np.where(patch == RED, 1, 0))

	# if numBlack + numBlue + numRed == 0:
	# 	return (WHITE)
	colors = [WHITE,BLACK, BLUE, RED]
	counts = [numWhite, numBlack, numBlue, numRed]
	colors = [BLACK, BLUE, RED]
	counts = [numBlack, numBlue, numRed]
	#print(counts)
	freq = {}
	for i in range(3):
		c = counts[i]
		color = colors[i]
		if c > 0:
			if c in freq:
				freq[c].append(color)
			else:
				freq[c] = [color]
	keyOrder = list(freq.keys())
	#print(freq)
	keyOrder.sort(reverse=True)
	order = []
	for k in keyOrder:
		for color in freq[k]:
			order.append(color)

	return tuple(order)

'''
We should only have a line going if there is a color
try removing white and seeing what is left
'''
def getColorFromOrder(colorOrder):
	colorOrder.remove(white)
	return list(colorOrder)[0]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Choose an input image')
	parser.add_argument('pathToImage', type=str, help='path to image of crease pattern')
	parser.add_argument('-t','--pathToThinned', help='path to thinned line image, if no path is specified, theline thinnng algorithm will run on the provided image')
	parser.add_argument('-g', '--grid', type=int, default=0, help='grid size of crease pattern, this will help in getting accurate points')
	parser.add_argument('-a', '--angle', type=float, default = 0, help='starting angle in degrees of crease pattern (try 22.5)')
	parser.add_argument('-sa', '--stepAngle', type=float, default=5.625, help='minimum angle in degrees between any two line segments. Selecting the largest possible step size will reduce the runtime')
	args = parser.parse_args()

	main(args)