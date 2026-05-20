#### Decision making - Autonomous systems - Real repository
- print: https://docs.google.com/document/d/1HHvDfez3BF-KAHr0Z_45H-KygH6nSYnAQoJ15cPIrlE/edit?usp=sharing

#### 20-MAY-2026
.65: not working
.52: not working 
.57 not working 
.64 not working 
.53 not working
.50 not working 
.67 not working 


##### Commands
##### Getting in the robot terminal SSH
1) ping 10.16.140.xx
2) ssh deec@10.16.140.xx
3) password: deecrobots

##### Transfering files (OPEN TERMINAL NOT INSIDE THE SSH, BUT NORMAL LINUX TERMINAL)
1) scp file.py deec@10.16.140.xx:~/
2) password: deecrobots

- in one robot terminal: ros2 launch alphabot2 alphabot2_launch.py
- in another robot terminal: ros2 run alphabot2 motion_driver
- in robot terminal: ros2 topic list

##### Moving robot
ros2 topic pub --rate 1 /alphabot2/cmd_vel geometry_msgs/msg/Twist "{linear: {x:
1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 5.0}}" ros2 run teleop_twist_keyboard teleop_twist_keyboard 

#### Topics to query
ros2 topic echo /image/compressed
#### To see the camera feed, run on the laptop/Lab computer
ros2 run rqt_image_view rqt_image_view

