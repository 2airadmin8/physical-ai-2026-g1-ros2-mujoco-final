# Physical AI 2026 Final Assignment — G1 MuJoCo / ROS2 v1.3.1

## Scope
Unitree G1 29DoF **rail-assisted** MuJoCo dynamics. NOT real robot, autonomous free biped walking, real camera perception, or safety-certified behavior. A horizontal rail force motor is added to a derived XML; the original G1 assets remain unmodified. Ramps and steps are visual fixtures with collision excluded in this rail benchmark. Synthetic 12-row training data is used for a logistic-regression GO/SLOW/STOP policy.

## Actual closed-loop path
- MuJoCo qpos/qvel and deterministic factory hazard --> /pai_final/sensor_observation (ROS2 JSON).
- Independent Python ROS AI policy process (pai_g1_ros_ai_policy.py) --> /pai_final/control_command.
- Actuator process checks random 128-bit session_id, awaited seq, phase, model SHA and commanded speed, rejects malformed / stale / duplicate responses.
- command --> d.ctrl rail force + 29 G1 joint PD actuators --> MuJoCo mj_step() physics.
- /pai_final/ai_decision and /pai_final/robot_feedback publish measured outcomes, including session_id.
- On AI timeout 4s, no more physics steps; this is a simulated fail-closed policy, not a certified safety stop.

## Version locks
- Unitree official upstream: https://github.com/unitreerobotics/unitree_mujoco
- Pinned commit: 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
- ROS2 Humble course image SHA: b58ec74c019aacd9ce4818fa25fc5a9e0a1f22f8c9cd1f5260994bf15cd9eb63
- MuJoCo 3.14.0, Ubuntu 22.04, ROS Python 3.10.
- Logistic policy SHA256: 9a606472b1daae2b9c892656be1a47adf329f9b1ad609910cc409d526a14839a.
- Simulator version v1.3.1 with 4-second ROS discovery warmup to ensure bag captures initial messages.

## Repeated verification (2026-10-30 JST)
- 3 x 72s routes; each final x = 12.84750 m.
- Each run: GO 494, SLOW 201, STOP 25; forklift STOP duration ~2.5s.
- Counterfactual at same initial x 9.550m: AI STOP x9.55603m vs forced GO x11.12148m after 3s; delta 1.56545m.
- rosbag2 v1.3.1: 2220 sensor observations; 2220 AI control commands; 2160 decisions; 2160 feedback = 8760 messages.
- Independent CDR decode: all command/observation sequences match, single session ID, no missing messages.
- 10 protocol unit tests PASS; false session/sequence, duplicate, malformed, SHA/phase/speed and timeout handled.
- A clean disposable ROS course image container re-provisioned successfully from pinned G1 and package versions; separate ROS AI process controlled the model in a fresh 5s test.
- Real MuJoCo 72.5s movie (2 frames/s visual sample) captured with v1.3.1 external ROS AI process. It includes rail-assist disclaimer.
- These are deterministic tests, **not** a validated statistical success rate.

## Key sources / data
- pai_g1_closed_loop_ros_external.py (v1.3.1)
- pai_g1_ros_ai_policy.py (separate independent ROS2 process)
- bootstrap_pai_mujoco.ps1 (pinned-model setup; runs only specified teaching container)
- test_ros_protocol_safety.py (10 tests)
- audit_rosbag_cdr.py (ROS2 CDR / session / sequence verification)
- pai_verify_ros_external.py (CSV/JSON/SQLite evidence checks)
- scene_factory_demo.xml (factory scene, must be installed beside official G1 MJCF)
- evidence/pai_v131_3runs (3x complete logs and ablation)
- evidence/rosbag2_v131 (4 topic SQLite)
- evidence/v131_rosbag_cdr_audit.json (strict audit)
- evidence/PAI_G1_ROS_AI_v131_72s.mp4 (v1.3.1 ROS footage)

## Restore and reproduce on 061
Extract ZIP, ensure image airobotbook/ros2-desktop-ai-robot-book-humble is running and Docker port 6080 does not collide. Windows PowerShell in extracted project folder:

1. Run bootstrap_pai_mujoco.ps1 -Container pai-ros2-course-061 -Root $PWD.
2. Start the AI process in the container:
   docker exec -d -u ubuntu pai-ros2-course-061 bash -lc 'source /opt/ros/humble/setup.bash; python3 /tmp/pai_g1_ros_ai_policy.py >/tmp/pai_policy.log 2>&1'
3. Run the simulation:
   docker exec -u ubuntu pai-ros2-course-061 bash -lc 'source /opt/ros/humble/setup.bash; python3 /tmp/pai_g1_closed_loop_ros_external.py --g1-dir /home/ubuntu/pai_closed_loop/g1 --out /tmp/pai_repro --repeats 3 --duration 72 --external-ros-ai'
4. Record 4 ROS topics with ros2 bag record before the simulator starts. Version v1.3.1 waits four seconds to allow subscriber discovery.
5. Verify CSV/JSON and rosbag with pai_verify_ros_external.py and audit_rosbag_cdr.py.

## Submission status / ethics
The technical simulator and AI closed-loop are validated. The Japanese PDF (draft with missing OmniCampus account ID), Public assignment-specific GitHub repo, YouTube unlisted-or-public playback URL, and OmniCampus submission receipt are separate requirements and not automatically complete.
No real names, emails, customer secrets, credentials or client data may be included in public artifacts.