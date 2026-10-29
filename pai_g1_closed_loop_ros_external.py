#!/usr/bin/env python3
"""PAI G1 closed-loop factory demo. Physics-actuated rail-assist testbed, NOT autonomous biped walking.

Unitree G1 29DoF is kept from an official pinned MuJoCo MJCF. For reproducible
factory-route experiments the floating pelvis joint is replaced in a DERIVED
model with a horizontal slide joint and a force motor (rail-assist rig).
The AI risk classifier directly regulates the actuator drive velocity. MuJoCo
integrates actual mass, friction, contact and joint torques with mj_step().
In the simulation loop we NEVER write d.qpos or d.qvel. A non-AI forced GO
ablation starts from the same state to prove causal influence.

Risk labels are synthetic instructional examples, NOT a certified safety model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

VERSION = "1.3.1"
MODEL_COMMIT = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
FEATURE_NAMES = ["aisle_clearance_m", "ramp_deg", "step_mm",
                 "human_or_vehicle", "low_light", "link_fault"]
TRAIN_X = np.asarray([
 [0.60,2,5,0,0,0],[0.40,4,8,0,0,0],[0.20,7,15,0,0,0],
 [0.15,7,15,1,0,0],[0.15,7,15,1,1,1],[0.50,3,5,1,0,0],
 [0.30,6,12,0,1,0],[0.80,1,2,0,0,0],[0.25,5,10,1,1,0],
 [0.70,2,4,0,1,1],[0.10,8,18,1,1,1],[0.45,5,10,0,0,1],
], dtype=np.float64)
TRAIN_Y = np.asarray([0,0,0,1,1,1,0,0,1,0,1,0], dtype=np.float64)
SCALE = np.asarray([1.,10.,20.,1.,1.,1.], dtype=np.float64)
SPEEDS = {"GO": 0.65, "SLOW": 0.35, "STOP": 0.0}


def trained_policy() -> dict[str, Any]:
    x = TRAIN_X / SCALE
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(1400):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -50, 50)))
        w -= 0.35 * (x.T @ (p - TRAIN_Y) / len(TRAIN_Y))
        b -= 0.35 * float(np.mean(p - TRAIN_Y))
    # Quantize for byte-identical weights across NumPy 1.x / 2.x reductions.
    w = np.round(w, 8)
    b = float(np.round(b, 8))
    digest = hashlib.sha256(np.r_[w, b].astype('<f8').tobytes()).hexdigest()
    return {"w": w, "bias": b, "sha256": digest, "samples": len(TRAIN_X)}


def ai_inference(policy: dict, features: np.ndarray) -> tuple[str, float]:
    z = float((features / SCALE) @ policy["w"] + policy["bias"])
    prob = float(1.0 / (1.0 + math.exp(-max(-40., min(40., z)))))
    if prob >= 0.60:
        return "STOP", prob
    if prob >= 0.012:
        return "SLOW", prob
    return "GO", prob


def build_scene(source_dir: Path, generated_dir: Path) -> Path:
    """Derive a rail-supported G1 physics fixture without overwriting official XML."""
    robot_path = source_dir / "g1_29dof.xml"
    scene_path = source_dir / "scene_factory_demo.xml"
    if not robot_path.is_file() or not scene_path.is_file():
        raise FileNotFoundError(f"Official G1 and factory XML required: {source_dir}")
    generated_dir.mkdir(parents=True, exist_ok=True)
    robot_out = generated_dir / "g1_29dof_pai_track.xml"
    scene_out = generated_dir / "scene_factory_ai_closed_loop.xml"
    mesh_link = generated_dir / "meshes"
    if not mesh_link.exists():
        mesh_link.symlink_to((source_dir / "meshes").resolve(), target_is_directory=True)
    root = ET.parse(robot_path)
    free = root.find(".//joint[@name='floating_base_joint']")
    if free is None:
        raise RuntimeError("Official floating G1 pelvis root joint not found")
    free.attrib.clear()
    free.attrib.update({
        "name": "pai_track_x", "type": "slide", "axis": "1 0 0",
        "limited": "true", "range": "-0.2 13.6",
        "damping": "6", "frictionloss": "0"
    })
    act = root.find("actuator")
    if act is None:
        raise RuntimeError("Original G1 torque actuators not found")
    ET.SubElement(act, "motor", {
        "name": "pai_ai_drive", "joint": "pai_track_x", "gear": "1",
        "ctrllimited": "true", "ctrlrange": "-600 600"
    })
    # The original body and all 29 motor actuators remain intact.
    root.write(robot_out, encoding="unicode", xml_declaration=True)
    txt = scene_path.read_text(encoding="utf-8")
    needle = '<include file="g1_29dof.xml"/>'
    if txt.count(needle) != 1:
        raise RuntimeError("Original scene include changed; refusing unsafe edit")
    # Rail-supported benchmark: ramp and step remain visible/sensed, but the
    # constrained horizontal rail does not physically climb them. Do not claim
    # this fixture validates independent G1 gait, slope or step traversal.
    stree = ET.ElementTree(ET.fromstring(txt.replace(
        needle, '<include file="g1_29dof_pai_track.xml"/>')))
    for geom_name in ("ramp_7deg", "step_15mm"):
        geom = stree.find(f".//geom[@name='{geom_name}']")
        if geom is None:
            raise RuntimeError(f"Missing terrain fixture: {geom_name}")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    vis = stree.find(".//visual/global")
    if vis is not None:
        vis.set("offwidth", "854")
        vis.set("offheight", "480")
    stree.write(scene_out, encoding="unicode", xml_declaration=True)
    return scene_out


class HazardSensor:
    """Deterministic scenario sensor. Perception feed is simulated, not a camera detector."""
    def __init__(self, forklift_hold_sec: float = 2.5):
        self.entered_at: float | None = None
        self.forklift_cleared = False
        self.hold = forklift_hold_sec

    def sense(self, x: float, t: float, ablation: bool = False) -> tuple[str, np.ndarray, bool]:
        hazard = np.asarray([0.15,7,15,1,1,1.])
        # Hysteresis: once the hazard is seen, a slight rearward drift CANNOT
        # remove it. Only a positive clearance event (hold interval elapsed)
        # releases STOP. This prevents unsafe GO/STOP chatter at x=9.4.
        if self.entered_at is not None and not self.forklift_cleared:
            if ablation or t - self.entered_at < self.hold:
                return "FORKLIFT_PRESENT", hazard, True
            self.forklift_cleared = True
        if x < 1.7:
            return "APPROACH", np.asarray([0.8,1,2,0,0,0.]), False
        if x < 7.0:
            return "NARROW_AISLE", np.asarray([0.32,1,3,0,0,0.]), False
        if x < 8.8:
            return "RAMP", np.asarray([0.35,7,15,0,0,0.]), False
        if x < 9.4:
            return "STEP", np.asarray([0.36,4,15,0,0,0.]), False
        if not self.forklift_cleared and x < 11.8:
            self.entered_at = t
            return "FORKLIFT_PRESENT", hazard, True
        if x >= 12.8:
            return "INSPECTION", np.asarray([0.8,0,0,0,0,0.]), False
        return "FORKLIFT_CLEAR", np.asarray([0.65,1,3,0,0,0.]), False


class RosBridge:
    def __init__(self, enabled: bool):
        self.node = None
        self.rclpy = None
        if enabled:
            import rclpy
            from std_msgs.msg import String
            self.String = String
            self.rclpy = rclpy
            rclpy.init()
            self.node = rclpy.create_node("pai_g1_mujoco_ai_closed_loop")
            self.decision_pub = self.node.create_publisher(String, "/pai_final/ai_decision", 10)
            self.feedback_pub = self.node.create_publisher(String, "/pai_final/robot_feedback", 10)
            print("ROS_TOPICS /pai_final/ai_decision /pai_final/robot_feedback", flush=True)

    def publish(self, row: dict) -> None:
        if self.node is None:
            return
        msg = self.String()
        payload = {k: row[k] for k in ("time_s", "phase", "risk_probability",
                                      "decision", "requested_speed_m_s")}
        if getattr(self, "session_id", None):
            payload["session_id"] = self.session_id
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.decision_pub.publish(msg)
        fb = self.String()
        payload = {k: row[k] for k in ("time_s", "base_x_m", "base_speed_m_s",
                                      "applied_drive_force_n", "right_shoulder_pitch_rad")}
        if getattr(self, "session_id", None):
            payload["session_id"] = self.session_id
        fb.data = json.dumps(payload, ensure_ascii=False)
        self.feedback_pub.publish(fb)
        self.rclpy.spin_once(self.node, timeout_sec=0)

    def close(self):
        if self.node:
            self.node.destroy_node()
            self.rclpy.shutdown()



class ExternalRosAI:
    """ROS-transported AI commands. No in-process risk inference in external mode."""
    def __init__(self, ros_bridge: RosBridge, policy_sha256: str, timeout_s: float = 4.0):
        if ros_bridge.node is None:
            raise RuntimeError("External ROS AI requires a ROS 2 node")
        self.ros=ros_bridge
        self.seq=0
        self.pending={}
        self.session_id=uuid.uuid4().hex
        self.awaiting_seq=None
        self.rejected_messages=0
        self.policy_sha256=policy_sha256
        self.timeout_s=timeout_s
        self.observation_pub=self.ros.node.create_publisher(
            self.ros.String, "/pai_final/sensor_observation", 10)
        self.command_sub=self.ros.node.create_subscription(
            self.ros.String, "/pai_final/control_command", self._on_command, 10)
        self.received=0
        print("EXTERNAL_ROS_AI_READY /pai_final/sensor_observation -> /pai_final/control_command",flush=True)

    def _on_command(self,msg):
        try:
            payload=json.loads(msg.data)
            seq=int(payload["seq"])
            if payload.get("session_id") != self.session_id or seq != self.awaiting_seq:
                self.rejected_messages += 1
                print("EXTERNAL_ROS_STALE_COMMAND_REJECTED",
                      "seq="+str(seq),"session="+str(payload.get("session_id")),flush=True)
                return
            if seq in self.pending:
                self.rejected_messages += 1
                print("EXTERNAL_ROS_DUPLICATE_COMMAND_REJECTED",seq,flush=True)
                return
            self.pending[seq]=payload
        except Exception as exc:
            self.rejected_messages += 1
            print("EXTERNAL_ROS_BAD_COMMAND",repr(exc),flush=True)

    def infer(self,features: np.ndarray, phase: str, x: float, t: float):
        self.seq+=1
        seq=self.seq
        self.awaiting_seq=seq
        obs=self.ros.String()
        obs.data=json.dumps({
            "seq":seq, "session_id":self.session_id, "features":[float(z) for z in features],
            "phase":phase, "robot_x_m":round(x,5),
            "sim_time_s":round(t,4)
        },separators=(",",":"))
        self.observation_pub.publish(obs)
        deadline=time.monotonic()+self.timeout_s
        while time.monotonic()<deadline:
            self.ros.rclpy.spin_once(self.ros.node,timeout_sec=0.04)
            if seq not in self.pending:
                continue
            command=self.pending.pop(seq)
            self.awaiting_seq=None
            if command.get("session_id")!=self.session_id or command.get("phase")!=phase:
                raise RuntimeError("ROS session/phase mismatch; rejecting command")
            if command.get("policy_sha256")!=self.policy_sha256:
                raise RuntimeError("ROS policy SHA mismatch; rejecting command")
            decision=str(command["decision"])
            risk=float(command["risk"])
            if decision not in SPEEDS or not math.isfinite(risk) or not 0.0<=risk<=1.0:
                raise RuntimeError("Invalid external ROS AI decision")
            if not math.isclose(float(command["requested_speed_m_s"]),SPEEDS[decision],abs_tol=1e-6):
                raise RuntimeError("ROS AI decision/speed mismatch")
            self.received+=1
            return decision, risk
        # Fail closed: no more physics steps if the AI command is missing.
        self.awaiting_seq=None
        raise TimeoutError(f"EXTERNAL_ROS_AI_TIMEOUT: seq={seq}, no independent AI control reply")


def do_run(model: mujoco.MjModel, policy: dict, output_dir: Path,
           *, scenario: str, seed: int, duration: float, ros: RosBridge | None = None,
           start_x: float | None = None, forced_go: bool = False,
           hazard_hold: float = 2.5, video: bool = False,
           video_fps: int = 2, video_camera: str = "customer_main",
           external_ai: ExternalRosAI | None = None) -> dict:
    # The rail fixture is constrained to horizontal translation.
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pai_track_x")
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pai_ai_drive")
    arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")
    if min(base_id, drive_id, arm_id) < 0:
        raise RuntimeError("Derived slide motor/arm joint not found")
    qadr = int(model.jnt_qposadr[base_id])
    vadr = int(model.jnt_dofadr[base_id])
    arm_qadr = int(model.jnt_qposadr[arm_id])
    # Only INITIALIZE qpos here. It must NEVER be written inside the control loop.
    if start_x is not None:
        data.qpos[qadr] = start_x
    mujoco.mj_forward(model, data)
    sensor = HazardSensor(forklift_hold_sec=hazard_hold)
    last_decision = ""
    counts = Counter()
    records: list[dict] = []
    control_every = max(1, round(0.10 / model.opt.timestep))
    n_steps = int(math.ceil(duration / model.opt.timestep))
    writer = None
    renderer = None
    frame_due = 0.0
    if video:
        try:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            import os
            os.environ.setdefault("MUJOCO_GL", "glfw")
            renderer = mujoco.Renderer(model, height=272, width=480)
            writer = imageio.get_writer(output_dir / f"{scenario}_{seed}.mp4", fps=video_fps, quality=7, macro_block_size=16)
            overlay_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
            print("VIDEO_RENDERER_READY", flush=True)
        except Exception as exc:
            print("VIDEO_RENDERER_UNAVAILABLE", repr(exc), flush=True)
            renderer = None
            writer = None
    t0 = time.monotonic()
    feedback_max = 0.0
    policy_time_total = 0.0
    stop_samples = 0
    stop_dwell = 0.0
    prev_v = 0.0
    try:
        for step in range(n_steps):
            t = float(data.time)
            x = float(data.qpos[qadr])
            v = float(data.qvel[vadr])
            phase, features, obstacle_active = sensor.sense(x, t, ablation=(scenario=="ablation"))
            if step % control_every == 0 or step == 0:
                ti = time.monotonic()
                ml_decision, risk = (external_ai.infer(features, phase, x, t)
                                     if external_ai else ai_inference(policy, features))
                policy_time_total += time.monotonic() - ti
                decision = "GO" if forced_go else ml_decision
                counts[decision] += 1
                if decision != last_decision:
                    print("DECISION", scenario, f"t={t:.2f}",f"x={x:.3f}",
                          f"{phase}","ml="+ml_decision,"action="+decision,
                          f"risk={risk:.3f}",flush=True)
                    last_decision = decision
                requested_speed = SPEEDS[decision]
                # Destination braking is independent of AI risk, a geometry safeguard.
                if x >= 12.90:
                    requested_speed = 0.0
                elif x > 12.45:
                    requested_speed = min(requested_speed, max(0.02, (12.90-x)*1.2))
                if decision == "STOP" and obstacle_active:
                    stop_samples += 1
                    stop_dwell += model.opt.timestep * control_every

                obs = {
                    "time_s": round(t,4),
                    "phase": phase,
                    "risk_probability": round(risk,5),
                    "decision": decision,
                    "model_decision": ml_decision,
                    "forklift_present": int(obstacle_active),
                    "requested_speed_m_s": round(requested_speed,4),
                    "base_x_m": round(x,5),
                    "base_speed_m_s": round(v,5),
                    "applied_drive_force_n": round(float(data.ctrl[drive_id]),4),
                    "right_shoulder_pitch_rad": round(float(data.qpos[arm_qadr]),5),
                }
                records.append(obs)
                if ros:
                    ros.publish(obs)
            # Physical drive velocity loop: motor force --> MuJoCo mass/contact dynamics.
            force = float(np.clip(340.0*(requested_speed-v), -580.0, 580.0))
            data.ctrl[drive_id] = force
            # Physical joint torque PD for 29 actuators: actual joint angles are sensed.
            # No direct qpos changes. All behavior passes through d.ctrl + mj_step.
            for i in range(model.nu):
                if i == drive_id:
                    continue
                jid = int(model.actuator_trnid[i][0])
                q = int(model.jnt_qposadr[jid])
                vj = int(model.jnt_dofadr[jid])
                target = 0.0
                jname = model.joint(jid).name
                if jname.endswith("knee_joint"):
                    target = 0.12
                if phase == "INSPECTION" and not obstacle_active and "right_shoulder_pitch" in jname:
                    target = -0.45
                if phase == "FORKLIFT_PRESENT" and "right_shoulder_pitch" in jname:
                    target = -0.20
                kp = 65.0 if "shoulder" not in jname and "wrist" not in jname else 20.0
                kd = 6.0 if kp > 30 else 1.8
                ctrl_range = model.actuator_ctrlrange[i]
                data.ctrl[i] = np.clip(kp * (target-data.qpos[q]) - kd*data.qvel[vj],
                                       float(ctrl_range[0]), float(ctrl_range[1]))
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), \
                f"Physics divergence at t={t}"
            feedback_max = max(feedback_max, abs(float(data.qvel[vadr])))
            if video and renderer is not None and float(data.time) >= frame_due:
                renderer.update_scene(data, camera=video_camera)
                img = Image.fromarray(renderer.render()).convert("RGB")
                canvas = ImageDraw.Draw(img)
                status_rgb = ((25,145,65) if decision=="GO" else
                              (190,140,20) if decision=="SLOW" else (190,45,45))
                canvas.rectangle((0,0,480,49),fill=(18,22,30))
                canvas.rectangle((0,0,9,49),fill=status_rgb)
                canvas.text((15,4),f"AI {decision}  risk={risk:.3f}   x={x:.2f} m",font=overlay_font,fill=(244,244,244))
                canvas.text((15,26),phase,font=overlay_font,fill=(190,210,235))
                canvas.rectangle((0,251,480,272),fill=(14,17,21))
                canvas.text((7,253),"G1 rail-assisted physics | not free walking",fill=(210,220,225))
                writer.append_data(np.asarray(img))
                frame_due += 1 / video_fps
            if x >= 12.95 and scenario != "ablation":
                print("GOAL_REACHED", scenario, f"x={x:.3f}",flush=True)
                break
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    fpath = output_dir / f"{scenario}_{seed}.csv"
    if records:
        with fpath.open("w", encoding="utf-8", newline="") as f:
            cw = csv.DictWriter(f, fieldnames=records[0].keys())
            cw.writeheader()
            cw.writerows(records)
    end_x = float(data.qpos[qadr])
    elapsed = time.monotonic() - t0
    summary = {
        "version": VERSION, "scenario": scenario, "seed": seed,
        "model_commit": MODEL_COMMIT,
        "risk_model_sha256": policy["sha256"], "num_training_examples": policy["samples"],
        "simulation_duration_s": round(float(data.time), 3),
        "wall_time_s": round(elapsed, 3), "steps": step+1,
        "end_x_m": round(end_x, 5),
        "max_abs_speed_m_s": round(feedback_max, 5),
        "physics_engine": mujoco.__version__,
        "control_path": ("ROS observation -> independent ROS ML policy -> ROS control command -> motor force -> MuJoCo mj_step -> feedback"
                         if external_ai else "logreg -> decision -> velocity setpoint -> force motor(ctrl) -> mj_step -> feedback"),
        "ai_command_over_ros": bool(external_ai),
        "ros_session_id": external_ai.session_id if external_ai else None,
        "ros_rejected_messages": external_ai.rejected_messages if external_ai else None,
        "go_count": counts["GO"], "slow_count": counts["SLOW"],
        "stop_count": counts["STOP"], "stop_seconds_approx": round(stop_dwell, 3),
        "start_x": float(start_x or 0.0), "forced_go": bool(forced_go),
        "trace_csv": str(fpath),
        "video_generated": bool(video and writer is not None),
        "max_feedback_speed_m_s": round(feedback_max,5)
    }
    (output_dir / f"{scenario}_{seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description="G1 AI->MuJoCo actuator->physics sensor closed loop")
    p.add_argument("--g1-dir",type=Path,required=True,help="pinned unitree_mujoco/unitree_robots/g1 dir")
    p.add_argument("--out",type=Path,default=Path("runs"))
    p.add_argument("--duration",type=float,default=72.0)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--time-step",type=float,default=0.005,help="physics dt (seconds)")
    p.add_argument("--smoke-only",action="store_true",help="short environment test without route completion assertions")
    p.add_argument("--ablation-only",action="store_true",help="test AI STOP vs forced GO at identical hazard initial condition")
    p.add_argument("--no-ablation",action="store_true")
    p.add_argument("--video",action="store_true")
    p.add_argument("--video-fps",type=int,default=2,choices=[1,2,3,4])
    p.add_argument("--video-camera",default="customer_main",choices=["customer_main","customer_side","customer_overview"])
    p.add_argument("--ros",action="store_true")
    p.add_argument("--external-ros-ai",action="store_true",help="receive AI control from independent ROS policy node; requires it to be running")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    scene=build_scene(a.g1_dir, a.out / "derived_model")
    model=mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = a.time_step
    print("SCENE_LOADED",str(scene),"nq",model.nq,"nu",model.nu,"timestep",model.opt.timestep,flush=True)
    assert model.nu==30, f"Expected 29 G1 torque motors + 1 rail motor; got {model.nu}"
    policy=trained_policy()
    print("POLICY",policy["sha256"],"training_examples",policy["samples"],flush=True)
    ros=RosBridge(a.ros or a.external_ros_ai)
    external_ai=ExternalRosAI(ros,policy["sha256"]) if a.external_ros_ai else None
    if external_ai:
        ros.session_id=external_ai.session_id
        # ROS discovery / rosbag recorder needs to subscribe before seq=1.
        # Without this delay the bag can miss initial messages even though
        # the simulator control loop itself is correct.
        print("ROS_DISCOVERY_WARMUP_4S", flush=True)
        time.sleep(4.0)
    all_summaries=[]
    try:
        for i in range(0 if a.ablation_only else a.repeats):
            all_summaries.append(do_run(model,policy,a.out,scenario="ai",
                            seed=i+1,duration=a.duration,ros=ros,video=a.video and i==0,
                            video_fps=a.video_fps,video_camera=a.video_camera,external_ai=external_ai))
        if a.ablation_only or (not a.no_ablation and not a.smoke_only):
            stop=do_run(model,policy,a.out,scenario="ablation_stop",
                        seed=0,duration=3.0,start_x=9.55,hazard_hold=99,external_ai=external_ai)
            forced=do_run(model,policy,a.out,scenario="ablation_forced_go",
                        seed=0,duration=3.0,start_x=9.55,forced_go=True,hazard_hold=99,external_ai=external_ai)
            differential=round(forced["end_x_m"]-stop["end_x_m"],5)
            result={"stop_end_x":stop["end_x_m"],"forced_go_end_x":forced["end_x_m"],
                    "causal_delta_m":differential, "ai_stop_count":stop["stop_count"],
                    "forced_go_count":forced["go_count"]}
            print("ABLATION",json.dumps(result),flush=True)
            (a.out/"counterfactual.json").write_text(json.dumps(result,indent=2)+"\n")
            if stop["stop_count"]==0 or differential < 0.10:
                raise AssertionError(f"AI command had insufficient causal motion effect: {result}")
        # End to end pass only if baseline reaches station & ablation shows impact.
        if not (a.smoke_only or a.ablation_only) and not all(s["end_x_m"]>12.80 for s in all_summaries):
            raise AssertionError("At least one baseline did not reach inspection station")
        if not (a.smoke_only or a.ablation_only) and not all(s["stop_count"]>0 for s in all_summaries):
            raise AssertionError("At least one baseline omitted forklift stop")
        (a.out/"all_runs.json").write_text(json.dumps(all_summaries,indent=2)+"\n")
        if external_ai:
            if external_ai.received < (20 if a.ablation_only else 50):
                raise AssertionError("Insufficient ROS AI command reception")
            print("EXTERNAL_ROS_AI_COMMANDS_ACCEPTED",external_ai.received,flush=True)
        print("PAI_ABLATION_ONLY_PASS" if a.ablation_only else "PAI_MUJOCO_AI_CLOSED_LOOP_PASS",len(all_summaries),"runs",flush=True)
    finally:
        ros.close()

if __name__=="__main__":
    main()