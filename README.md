# ros\_text\_detector

ros\_text\_detector is a ROS package that can detect text in images from robot cameras. By default, it subscribes to /camera/rgb/image\_raw to obtain images. It then uses openCV and EAST text detection software to find text. Finally, the program will publish the coordinates of the bounding box(es) to /counter. 

### Prerequisites
- This repo currently works on python2.7.
- You must also have _at least_ openCV 3.4.2 (or openCV 4) installed on your system for EAST text detection to work. 
- Having an updated version of imutils is also necessary. On Linux, install imutils by running `pip install --upgrade imutils`.

### Usage
Install the package from this repo by downloading it to your catkin workspace and running `catkin_make`, followed by `source devel/setup.bash`.

Go to the scripts folder and open image\_text\_detection.py. Change `east_path_file` (line 30) to the correct location of the frozen\_east\_text\_detection.pb file in your system. 

To run the program, run `roslaunch ros_text_detector text_detection.launch`

- If you wish to change the publisher or subscriber topic, open image\_text\_detection.py. The ROS publisher is found on line 21, and the ROS subscriber is found on line 163.
### Performance
Depending on the system running the program, text detection of each frame may take anywhere from 0.5 to 1.5 seconds. On a 1.5 second system, the program and video display will be delayed by approximately 10 seconds (e.g. a movement of the robot camera will show up on the video display ~10 seconds after the movement occurs). 



### Acknowledgements
- The code builds upon Adrian Rosebrick's EAST text detection program. His article on his program can be found [here](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/).
- [This article](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/) may help with the installation of openCV. 

