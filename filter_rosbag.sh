#!/bin/bash
src_bag=$1
dst_bag=$2
source /opt/ros/noetic/setup.bash
rosbag filter $src_bag $dst_bag "topic.startswith('/sensing/gnss') or topic.startswith('/sensing/imu') or topic.startswith('/sensing/lidar') or topic.startswith('/vehicle') or topic == '/vehicle' or topic == '/tf_static'"