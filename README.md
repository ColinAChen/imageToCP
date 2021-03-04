# imageToCP
Take an image of a crease pattern and create a .cp file of it.
## Usage
Requires Python, OpenCV, and Numpy.

`python imageToCP path\to\image`

The crease pattern will get saved as the image name with a .cp extension.
## Motivation
Origami crease patterns can be used share origami designs. While an image of a crease pattern is sufficient for an origami artist to recreate a model, it is insufficient for origami software to recreate a model. Being able to recreate a crease pattern file from an image would allow artists to work on any crease pattern using origami software.

## Methods
My approach was to use image processing to find lines in the input image. This step was accomplished using a Hough Line Transform implemented with [OpenCV](https://www.google.com/search?q=hough+line+transform&rlz=1C1JZAP_enUS816US817&oq=hough+line+transform&aqs=chrome..69i57.3010j0j1&sourceid=chrome&ie=UTF-8).
This approach is image file type agnostic so any image file that OpenCV can parse will work. In the future, it should allow users to create crease pattern files from camera images of crease patterns.


## Future work
There are main improvements that need to be implemented before this approach is practical.
#### Image preprocessing
I would like to perform image preprocessing to ensure images are uniform before perform line detection. Right now, the lines I worked were thick enough for the line detection algorithm to double count them which results in lost precision. Performing preprocessing should remove this problem and ensure no information is lost.
#### Determining line bounds
Hough Line Transform caluclates lines parametrically and doesn't record the bounds of lines. Once lines are determined, there needs to be another pass through the image to determine where line bounds are located before the final crease pattern can be written.
