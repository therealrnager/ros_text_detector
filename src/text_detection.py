#! /usr/bin/env python


# USAGE: run this in command line. make sure to cd into the folder first
# python text_detection.py

# import ROS packages first
import rospy
import roslib
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') #necessary so that cv2 version 4.2.0 imports, not sure why it doesn't work if this isn't here
import cv2
print("[cv2 version]: " + cv2.__version__)
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('text_detection')
pub = rospy.Publisher('/counter', Int32MultiArray, queue_size=1)
foo = Int32MultiArray()
foo.data = [1, 2, 3 ,4]

#import other packages needed
from imutils.object_detection import non_max_suppression
import numpy as np
import time

east_path_file = "/home/turtlebot/catkin_ws/src/ros_text_detector/src/frozen_east_text_detection.pb"

#create cvBridge to help convert image to an image compatible with openCV
bridge_object = CvBridge()

def camera_callback(data):
	try:
		# We select bgr8 because its the OpneCV encoding by default
		cv_image = bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
	except CvBridgeError as e:		
		(e)


	# load the input image and grab the image dimensions
	initial_image = cv_image
	orig = initial_image.copy()
	crop_edge_x = 320
	crop_edge_y = 180
	image = orig_image[crop_edge_y:720 - crop_edge_y,crop_edge_x:1280 - crop_edge_x]
	(H, W) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	#(newW, newH) = (args["width"], args["height"])
	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	#net = cv2.dnn.readNet(args["east"])
	net = cv2.dnn.readNet(east_path_file)

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()

	# show timing information on text prediction
	print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.2:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the image
		cv2.rectangle(orig, (crop_edge_x + startX, crop_edge_y + startY), (crop_edge_x + endX, crop_edge_y + endY), (0, 255, 0), 2)

		#publish bounding box coordinates
		foo.data =[crop_edge_x + startX, crop_edge_y + startY, crop_edge_x + endX, crop_edge_y + endY]
		pub.publish(foo)


	# show the output image
	cv2.imshow("Text Detection", orig)
	cv2.waitKey(1)
	#cv2.destroyAllWindows()

#not sure why queue_size isn't working or throttling incoming messages, which is what I want to happen
#solved: many people have similar issue, must add buff_size. this reduces delay from 50s --> 9s
image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, camera_callback,queue_size=1,buff_size=2**25)


try:
	rospy.spin()
	
	#print("hi")
except KeyboardInterrupt:
	print("Shutting down")

cv2.destroyAllWindows()
