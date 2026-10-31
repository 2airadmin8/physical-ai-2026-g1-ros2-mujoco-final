#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
export DISPLAY=:0 MUJOCO_GL=glfw
root=/tmp/pai_c81_final
mkdir -p "$root"
if [ -e "$root/rosbag2" ]; then echo C81_OLD_BAG_REFUSE; exit 3; fi
ros2 bag record -o "$root/rosbag2" /pai_b/object_features /pai_b/ai_classification /pai_b/robot_state /pai_b/phase > "$root/bag.log" 2>&1 &
bg=$!
cleanup(){ kill -INT "$bg" 2>/dev/null || true; wait "$bg" 2>/dev/null || true; }
trap cleanup EXIT
sleep 3
for i in 0 1 2; do
  export PAI_C81_LAYOUT="$i"
  echo C81_FINAL_ROS_LAYOUT="$i"
  mkdir -p "$root/layout_$i"
  cd /tmp
  python3 pai_c81_robot_runtime.py --out "$root/layout_$i" \
    --video --video-fps 6 --duration 100 2>&1 | tee "$root/layout_$i/run.log"
done
printf "file '%s'\n" "$root/layout_0/PAI_C81_AI_ROS_PHYSICS_PICK.mp4" \
 "$root/layout_1/PAI_C81_AI_ROS_PHYSICS_PICK.mp4" \
 "$root/layout_2/PAI_C81_AI_ROS_PHYSICS_PICK.mp4" > "$root/concat.txt"
ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$root/concat.txt" \
 -c copy -movflags +faststart "$root/PAI_C81_FINAL_3LAYOUTS_85s.mp4"
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 \
 "$root/PAI_C81_FINAL_3LAYOUTS_85s.mp4"
echo "C81_3LAYOUTS_ROS_VIDEO_PASS"