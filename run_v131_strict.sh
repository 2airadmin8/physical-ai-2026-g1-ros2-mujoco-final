#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root=/tmp/pai_v131_release
bag=/tmp/pai_v131_rosbag
mkdir -p "$root"
ros2 bag record -o "$bag" \
  /pai_final/sensor_observation /pai_final/control_command \
  /pai_final/ai_decision /pai_final/robot_feedback > /tmp/pai_v131_bag.log 2>&1 &
pid=$!
stop_recorder(){
  kill -INT "$pid" 2>/dev/null || true
  wait "$pid" || true
}
trap stop_recorder EXIT
sleep 3
python3 /tmp/pai_g1_closed_loop_ros_external.py \
  --g1-dir /home/ubuntu/pai_closed_loop/g1 --out "$root" \
  --duration 72 --repeats 3 --external-ros-ai --time-step 0.005
echo PAI_V131_FULL_ROS_PASS