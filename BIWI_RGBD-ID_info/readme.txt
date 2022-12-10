### DESCRIPTION
The BIWI RGBD-ID Dataset is a RGB-D dataset of people targeted to long-term people re-identification from RGB-D cameras.

It contains 50 training and 56 testing sequences of 50 different people. The dataset includes synchronized RGB images (captured at the highest resolution possible with a Microsoft Kinect for Windows, i.e. 1280x960 pixels), depth images, persons' segmentation maps and skeletal data (as provided by Microsoft Kinect SDK), in addition to the ground plane coordinates. These videos have been acquired at about 10fps.
In the training videos, people performs a certain routine of motions in front of a Kinect, such as a rotation around the vertical axis, several head movements and two walks towards the camera.
28 people out of 50 present in the training set have been recorded also in two testing videos each. These testing sequences have been collected in a different day and in a different location with respect to the training dataset, therefore most subjects are dressed differently. For every person in the testing set, a Still sequence and a Walking sequence have been collected. In the Still video, every person is still or slightly moving in place, while in the Walking video, every person performs two walks frontally and two other walks diagonally with respect to the Kinect.


## References:

If you use the BIWI RGBD-ID dataset, please cite the following work:

M. Munaro, A. Fossati, A. Basso, E. Menegatti and L. Van Gool.
"One-Shot Person Re-Identification with a Consumer Depth Camera"
Book Chapter in "Person Re-Identification", Springer, 2013.


## Data description:

All the samples in the BIWI RGBD-ID Dataset are provided as folders with 5 different files for every frame:
- rgb image: 1280x960 resolution
- depth image: 640x480 resolution, 16bit, with depth values in millimeters
- user map: 640x480 resolution, 8bit, with user id labels at every pixel (0 means 'no person')
- txt file with (a,b,c,d) ground plane coefficients (in ax + by + cz + d = 0 form)
- txt file with skeleton tracker joint position and links orientation (estimated with Microsoft Kinect SDK and referred to the depth reference frame).
 

## Structure of the skeleton files:

For every frame, a skeleton file is available. For every joint, a row with the following information is written to the skeleton file:
[id]: person ID,
[x3D], [y3D], [z3D]: joint 3D position with respect to the Kinect depth frame,
[x2D], [y2D]: joint 2D position in the depth image,
[TrackingState]: 0 (not tracked), 1 (inferred) or 2 (tracked),
[QualityFlag]: if > 0, the person is partially out of the image (corresponds to skeleton.dwQualityFlags),
[OrientationStartJoint], [OrientationEndJoint]: indices of the extreme joints of a link,
[Qx], [Qy], [Qz], [Qw]: quaternion expressing the orientation of the link identified by [OrientationStartJoint] and [OrientationEndJoint].
For more information, please visit http://msdn.microsoft.com/en-us/library/hh973074.aspx.


## Kinect calibration parameters:

RGB intrinsics matrix: [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
Depth intrinsics matrix: [575.8, 0.0, 319.5, 0.0, 575.8, 239.5, 0.0, 0.0, 1.0]
RGB-Depth extrinsic parameters: T = [0.025 0.0 0.0], R = [0.0 0.0 0.0].


## Copyright:

This work is licensed under a Creative Commons Attribution-Noncommercial-Share Alike 3.0 Unported License. 
This dataset is distributed WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


For questions and remarks directly related to the BIWI RGBD-ID dataset please contact  matteo.munaro@dei.unipd.it
 