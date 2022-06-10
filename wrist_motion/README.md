# wrist_motion

Sub-package that takes care of motion planning for wrist reorientation during data collection.

After planning, we publish markers for:
    * position constraint in `base_link` (box marker in front of robot)
    * start gripper frame and final gripper frame (at the end of the calculated trajectory)
    * three arrows:
      * blue indicates the z axis' starting orientation
      * black is the sampled / desired orientation
      * white is the orientation we actually reach after executing the trajectory
    * finally, we also publish the trajectory itself

This package comes with a launch file for TIAGo to test the reorientation in simulation (assumes various TIAGo packages are installed, e.g. description).
An rviz config file that displays the markers published by `Reorient` is also included.
