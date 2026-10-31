#!/usr/bin/env bash
# Reproducible educational simulator run, independent of any pre-existing ROS AI node.
set -eo pipefail
source /opt/ros/humble/setup.bash
export DISPLAY=:0 MUJOCO_GL=glfw
export ROS_DOMAIN_ID=77
script_dir="$(cd "$(dirname "$0")" && pwd)"
if [ -n "$PAI_C81_OUT_ROOT" ]; then
  root="$PAI_C81_OUT_ROOT"
else
  root=/tmp/pai_c81_v3_isolated_qa
fi
mkdir -p "$root"
if [ -e "$root/rosbag2" ]; then echo OUTPUT_EXISTS_REFUSE; exit 3; fi
cd "$script_dir"
python3 "$script_dir/pai_c81_ai_node.py" > "$root/ai_node.log" 2>&1 &
ai_pid=$!
bag_pid=""
cleanup(){
  if [ -n "$bag_pid" ]; then kill -INT "$bag_pid" 2>/dev/null || true; wait "$bag_pid" 2>/dev/null || true; fi
  kill -INT "$ai_pid" 2>/dev/null || true
  wait "$ai_pid" 2>/dev/null || true
}
trap cleanup EXIT
sleep 3
if ! kill -0 "$ai_pid" 2>/dev/null; then echo AI_NODE_EXITED; exit 4; fi
ros2 bag record -o "$root/rosbag2" /pai_b/object_features /pai_b/ai_classification /pai_b/robot_state /pai_b/phase > "$root/bag.log" 2>&1 &
bag_pid=$!
sleep 4
if ! kill -0 "$bag_pid" 2>/dev/null; then echo BAG_RECORDER_EXITED; exit 5; fi
for i in 0 1 2; do
  export PAI_C81_LAYOUT="$i"
  mkdir -p "$root/layout_$i"
  echo V3_ISOLATED_LAYOUT="$i" ROS_DOMAIN_ID="$ROS_DOMAIN_ID"
  python3 "$script_dir/pai_c81_robot_runtime.py" --out "$root/layout_$i" --video --video-fps 6 --duration 100 2>&1 | tee "$root/layout_$i/run.log"
done
printf "file '%s'\n" "$root/layout_0/PAI_C81_AI_ROS_PHYSICS_PICK.mp4" "$root/layout_1/PAI_C81_AI_ROS_PHYSICS_PICK.mp4" "$root/layout_2/PAI_C81_AI_ROS_PHYSICS_PICK.mp4" > "$root/concat.txt"
ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$root/concat.txt" -c copy -movflags +faststart "$root/PAI_C81_FINAL_3LAYOUTS_85s.mp4"
kill -INT "$bag_pid" 2>/dev/null || true
wait "$bag_pid" 2>/dev/null || true
bag_pid=""
python3 "$script_dir/pai_c81_independent_qa.py" --root "$root"
(cd "$script_dir" && python3 -m unittest test_pai_c81_protocol -v)
echo C81_ISOLATED_FRESH_REPRODUCTION_PASS
