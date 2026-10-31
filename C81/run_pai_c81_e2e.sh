#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
set -u
export DISPLAY=:0
export MUJOCO_GL=glfw
mkdir -p /tmp/pai_c81_e2e
bag=/tmp/pai_c81_e2e/rosbag2
ros2 bag record -o "$bag" \
 /pai_b/object_features /pai_b/ai_classification \
 /pai_b/robot_state /pai_b/phase >/tmp/pai_c81_bag.log 2>&1 &
pid=$!
function quit() { kill -INT "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
trap quit EXIT
sleep 3
cd /tmp
python3 pai_c81_robot_runtime.py --out /tmp/pai_c81_e2e --video --video-fps 4 --duration 80
echo C81_EXTERNAL_ROS_AND_RENDER_DONE