# Lucas Kanade Template Tracker

## Introduction 

The Lucas Kanade Tracker is used in computer vision applications as a differential method for estimating optical flow using brightness constancy. This method
works based on the assumption of constant flow in the local neighborhood of the pixel and solving the optical flow equations for all the neighboring pixels using
least squares criterion.

As a part of the implementation of the Lucas Kanade algorithm, we have tested the outputs on three different videos.

The outputs for each video are:
- The first video of tracking Bolt is in ``outputs/Bolt_Tracker_Output.avi``
- The second video of tracking a car is in ``outputs/Car_Tracker_Output.avi``
- The third video of tracking a baby is in ``outputs/Baby_Tracker_Output.avi``


The report containing the details of the outputs and the plots is [here](https://github.com/kmushty/Lucas_Kanade_Tracker/blob/main/Report.pdf) 


## Dependencies

The following are the project dependencies:
- OpenCV 3.4.2 or above
- Python 3.5

## Code Execution

In order to implement the code:
- Clone the repo
- Run using the following command in the command line ``python3 GMM.py``
- After running the code, select the bounding box from the top left to the bottom right and press enter. Then, the outputs of each frame is shown along with the written video.

## Output 

<img src="LKtracker1.png" width="640" height="480">

<img src="LKtracker2.png" width="640" height="480">

<img src="LKtracker3.png" width="640" height="480">
