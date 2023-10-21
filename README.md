# Object-Pose-Estimation-and-Grasping

In the dynamic realm of robotics, achieving precision in object recognition and manipulation is paramount. This project  employs deep learning techniques to enhance computer vision for robotic grasping. Here are the key highlights:

1. Enhanced Object Recognition Accuracy:
To facilitate precise robotic grasping, I engineered a segmentation model using a simplified U-Net neural network, implemented in PyTorch. This model predicted masks from RGB images, significantly improving object recognition accuracy. This laid the foundation for more accurate and reliable robotic interactions with the environment

2. 3D World Understanding:
To empower the robot with a comprehensive perception of its surroundings, I generated object point clouds from RGBD images. This involved the generation of segmentation masks and depth masks, followed by the extraction of world-coordinate point clouds. This approach enabled the robot to perceive objects in three dimensions, facilitating more informed decision-making during grasping

3. Precise Object Positioning and Orientation:
The success of robotic grasping hinges on accurate object positioning and orientation. Here, I skillfully applied the Iterative Closest Point (ICP) algorithm to align the original and segmented point clouds with precision. This alignment not only ensured accurate object detection but also enabled the robot to determine the optimal positioning and orientation for a successful grasp

