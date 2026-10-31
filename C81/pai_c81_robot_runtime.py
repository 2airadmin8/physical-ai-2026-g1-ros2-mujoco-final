#!/usr/bin/env python3
"""C81 end-to-end AI -> ROS 2 -> force-driven MuJoCo cart & tool -> contact-only
vacuum grasp -> transport -> release into bin -> reinspection.
Scope: simulated color+shape sensor (not camera), simplified holonomic base
(force-driven planar Cartesian proxy), 3-axis manipulator (not six-axis arm).
Suction actuates a weld ONLY upon measured MuJoCo geom contact, updates its
current relative offset, and deactivates after reaching the bin. Never set free
object qpos/qvel after initialization or teleport an object.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, time, uuid
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco
from pai_c81_mujoco import xml_scene,OBJECTS,KINDS,BIN
from pai_c81_ai_node import classify,SHA,PROTOTYPES

VERSION="0.2.0"
class Bridge:
    def __init__(self,use_ros,timeout=4):
        self.node=None;self.rclpy=None;self.use_ros=use_ros
        self.session_id=uuid.uuid4().hex
        self.pending={}
        self.awaiting=None
        self.seq=0
        self.received=0
        self.rejected=0
        self.timeout=timeout
        if use_ros:
            import rclpy
            from std_msgs.msg import String
            self.rclpy=rclpy;self.String=String
            rclpy.init()
            self.node=rclpy.create_node("pai_c81_physics_pick_receiver")
            self.obs=self.node.create_publisher(String,"/pai_b/object_features",10)
            self.state_pub=self.node.create_publisher(String,"/pai_b/robot_state",10)
            self.phase_pub=self.node.create_publisher(String,"/pai_b/phase",10)
            self.node.create_subscription(String,"/pai_b/ai_classification",self.receive,10)
            print("C81_ROS_READY",self.session_id,flush=True)
            time.sleep(3.0) # allow rosbag and independent AI subscriber discovery
    def receive(self,msg):
        try:
            a=json.loads(msg.data)
            seq=int(a["seq"])
            if a.get("session_id") != self.session_id or seq!=self.awaiting or seq in self.pending:
                self.rejected+=1
                print("C81_REJECT_STALE_OR_DUPLICATE",seq,flush=True)
                return
            self.pending[seq]=a
        except Exception as e:self.rejected+=1;print("C81_BAD_ROS",repr(e),flush=True)
    def ai(self,features,objid,px,py,phase):
        self.seq+=1;seq=self.seq;self.awaiting=seq
        if not self.use_ros:
            label,confidence=classify(features)
            self.awaiting=None
            return {"session_id":self.session_id,"seq":seq,"label":label,
                    "confidence":confidence,"policy_sha256":SHA,
                    "safety":"SKIP" if label=="personal" or confidence<.75 else "PICK"}
        msg=self.String()
        msg.data=json.dumps({"session_id":self.session_id,"seq":seq,
             "features":features,"object_id":objid,"observed_x_m":px,
             "observed_y_m":py,"phase":phase},separators=(",",":"))
        self.obs.publish(msg)
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            self.rclpy.spin_once(self.node,timeout_sec=.04)
            if seq not in self.pending:continue
            a=self.pending.pop(seq)
            self.awaiting=None
            if a.get("policy_sha256")!=SHA:
                raise RuntimeError("POLICY_SHA_FAIL_CLOSED")
            if a.get("label") not in ("paper","cup","wrapper","personal") or not 0<=float(a["confidence"])<=1:
                raise RuntimeError("INVALID_AI_COMMAND_FAIL_CLOSED")
            if a.get("safety")!=("SKIP" if a["label"]=="personal" or float(a["confidence"])<.75 else "PICK"):
                raise RuntimeError("AI_SAFETY_MISMATCH_FAIL_CLOSED")
            self.received+=1
            return a
        self.awaiting=None
        raise TimeoutError("C81_EXTERNAL_AI_TIMEOUT_FAIL_CLOSED seq="+str(seq))
    def emit(self,publisher,payload):
        if self.node:
            msg=self.String();msg.data=json.dumps(payload,separators=(",",":"))
            publisher.publish(msg)
            self.rclpy.spin_once(self.node,timeout_sec=0)
    def close(self):
        if self.node:self.node.destroy_node();self.rclpy.shutdown()

def model_ids(m):
    j={}
    for name in ("cart_x","cart_y","reach_x","reach_y","lift"):
        i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        j[name]=(int(m.jnt_qposadr[i]),int(m.jnt_dofadr[i]))
    return j
def set_weld(m,d,index,tool_body,obj_body):
    # Compute relative body pose at contact, rather than reverting to the
    # initial model pose, which would cause a huge physically false impulse.
    pos=d.xpos[obj_body]-d.xpos[tool_body]
    inv=d.xquat[tool_body].copy();inv[1:]*=-1
    rot=np.empty(9);mujoco.mju_quat2Mat(rot,inv)
    relpos=rot.reshape(3,3)@pos
    q=np.empty(4)
    mujoco.mju_mulQuat(q,inv,d.xquat[obj_body])
    m.eq_data[index,3:6]=relpos
    m.eq_data[index,6:10]=q
    d.eq_active[index]=1
    mujoco.mj_forward(m,d)
def contact_with(m,d,geomA,geomB):
    for i in range(d.ncon):
        c=d.contact[i]
        if (c.geom1==geomA and c.geom2==geomB) or (c.geom1==geomB and c.geom2==geomA):
            return True
    return False

def do_run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_string(xml_scene())
    model.opt.timestep=.005
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    (args.out/"pai_c81_model.xml").write_text(xml_scene(),encoding="utf-8")
    ids=model_ids(model)
    arm_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"suction")
    gripper=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"tool")
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"suction_site")
    objs={}
    for name,kind,xy in OBJECTS:
        objs[name]={
            "kind":kind,"initial_xy":xy,
            "body":mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name),
            "geom":mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"g_"+name),
            "weld":mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,"vacuum_"+name),
        }
    print("C81_SCENE_VALIDATED nq",model.nq,"nu",model.nu,"neq",model.neq,flush=True)
    bridge=Bridge(not args.no_ros)
    phase="SCAN"
    start_stage=0.0
    current=None
    current_kind=None
    attempts=0
    accepted=[]
    skipped=[]
    labels=[]
    failed=[]
    log=[]
    control=dict(zip(ids,[0.0,0.0,.48,0.0,0.0]))
    base_goal=[-1.4,0]
    frame_due=0.0
    frame_count=0
    video_writer=None;renderer=None
    viewer=None
    if args.video:
        try:
            os.environ.setdefault("MUJOCO_GL","glfw")
            import imageio.v2 as imageio
            from PIL import Image,ImageDraw,ImageFont
            renderer=mujoco.Renderer(model,height=480,width=848)
            video_writer=imageio.get_writer(args.out/"PAI_C81_AI_ROS_PHYSICS_PICK.mp4",
                  fps=args.video_fps,quality=7,macro_block_size=16)
            font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
            print("C81_VIDEO_READY",flush=True)
        except Exception as e:print("C81_VIDEO_UNAVAILABLE",repr(e),flush=True)
    if args.viewer:
        try:
            from mujoco import viewer as mjviewer
            viewer=mjviewer.launch_passive(model,data)
            print("C81_INTERACTIVE_VIEWER_READY",flush=True)
        except Exception as e:print("C81_VIEWER_UNAVAILABLE",repr(e),flush=True)
    def to_phase(name):
        nonlocal phase,start_stage
        phase=name;start_stage=float(data.time)
        print("C81_PHASE",name,"t",round(float(data.time),2),
              "item",current,"cart",tuple(round(x,2) for x in base_pos()),flush=True)
    def base_pos():
        return (float(data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"cart"),0]),
                float(data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"cart"),1]))
    def near_base():
        bx,by=base_pos()
        vx=float(data.qvel[ids["cart_x"][1]]);vy=float(data.qvel[ids["cart_y"][1]])
        return abs(bx-base_goal[0])<.065 and abs(by-base_goal[1])<.065 and math.hypot(vx,vy)<.12
    def pt(obj):
        return data.xpos[obj["body"]]
    active_weld=None
    start_wall=time.monotonic()
    steps=int(args.duration/model.opt.timestep)
    try:
        for step in range(steps):
            t=float(data.time)
            if phase=="SCAN":
                candidates=[name for name,_,_ in OBJECTS if name not in accepted and name not in skipped and name not in failed]
                if not candidates:
                    phase="DONE"
                    print("C81_TASKS_FINISHED",len(accepted),len(skipped),len(failed),flush=True)
                else:
                    current=candidates[0]
                    obj=objs[current]
                    k=obj["kind"]
                    color,size,_=KINDS[k]
                    x=pt(obj)
                    rgb=[float(z) for z in color.split()[:3]]
                    dim=[float(z) for z in size.split()]
                    # Appearance + dimension descriptor simulated from geometry.
                    feature=rgb+dim
                    ai=bridge.ai(feature,current,round(float(x[0]),4),round(float(x[1]),4),"SCAN")
                    pred=ai["label"]; labels.append({"id":current,"pred":pred,"truth":k,
                      "confidence":ai["confidence"],"safety":ai["safety"]})
                    print("AI_OBSERVATION",current,pred,ai["confidence"],ai["safety"],flush=True)
                    if ai["safety"]=="SKIP":
                        skipped.append(current)
                        if k!="personal":failed.append(current)
                        current=None
                        continue
                    if pred!=k:failed.append(current);current=None;continue
                    current_kind=k
                    x=pt(obj)
                    base_goal=[float(x[0])-.48,float(x[1])]
                    control["reach_x"]=.48;control["reach_y"]=0;control["lift"]=0.0
                    to_phase("NAVIGATE_TO_LITTER")
            elif phase=="NAVIGATE_TO_LITTER":
                x=pt(objs[current]);base_goal=[float(x[0])-.48,float(x[1])]
                if near_base() and abs(float(data.qpos[ids["reach_x"][0]])-.48)<.04:
                    to_phase("LOWER_TOOL")
            elif phase=="LOWER_TOOL":
                obj=objs[current];pos=pt(obj)
                zhalf=float(model.geom_size[obj["geom"],1] if current_kind=="cup" else model.geom_size[obj["geom"],2])
                top=float(pos[2])+zhalf
                # Unloaded cup center = .68m at lift=0. Bottom of sphere
                # touches when sphere center is object top + sphere radius.
                control["lift"]=float(np.clip(top+.037-.68,-.73,0))
                if contact_with(model,data,arm_geom,obj["geom"]):
                    set_weld(model,data,obj["weld"],gripper,obj["body"])
                    active_weld=obj["weld"]
                    print("C81_PHYSICAL_CONTACT_ATTACH",current,
                          "t",round(t,3),"obj_z",round(float(pos[2]),3),flush=True)
                    control["lift"]=0
                    to_phase("LIFT_VERIFICATION")
                elif t-start_stage>8.0:
                    failed.append(current)
                    print("C81_PICK_CONTACT_TIMEOUT",current,flush=True)
                    control["lift"]=0
                    to_phase("RESET")
            elif phase=="LIFT_VERIFICATION":
                z=float(pt(objs[current])[2])
                if z>.47 and abs(float(data.qvel[ids["lift"][1]]))<.2:
                    print("C81_LIFTED",current,"z",round(z,3),flush=True)
                    base_goal=[BIN[0]-.60,BIN[1]]
                    to_phase("TRANSPORT_TO_BIN")
                elif t-start_stage>6:
                    failed.append(current)
                    to_phase("RESET")
            elif phase=="TRANSPORT_TO_BIN":
                if near_base():
                    control["reach_x"]=.60
                    if abs(float(data.qpos[ids["reach_x"][0]])-.60)<.04:
                        if active_weld is None:raise RuntimeError("LOST_SUCTION_FAIL_CLOSED")
                        data.eq_active[active_weld]=0
                        print("C81_RELEASE_ABOVE_BIN",current,
                              "xyz",[round(float(v),3) for v in pt(objs[current])],flush=True)
                        active_weld=None
                        to_phase("DROP_REINSPECT")
            elif phase=="DROP_REINSPECT":
                if t-start_stage>1.4:
                    p=pt(objs[current]);x,y,z=[float(x) for x in p]
                    in_bin=(abs(x-BIN[0])<.25 and abs(y-BIN[1])<.25 and .06<z<.45)
                    if in_bin:
                        accepted.append(current)
                        print("C81_PHYSICALLY_COLLECTED",current,"xyz",tuple(round(q,3) for q in (x,y,z)),flush=True)
                    else:
                        failed.append(current)
                        print("C81_BIN_VERIFICATION_FAIL",current,"xyz",(x,y,z),flush=True)
                    current=None
                    to_phase("SCAN")
            elif phase=="RESET":
                if active_weld is not None:
                    data.eq_active[active_weld]=0;active_weld=None
                if t-start_stage>1:
                    current=None;to_phase("SCAN")
            elif phase=="DONE":
                # keep final animation to show result
                if t-start_stage>3.0:break
            # Dynamics: actuated joint forces only. Cart is a holonomic
            # platform drive abstraction, arm is 3-axis gantry, NO qpos teleports.
            target={
                "cart_x":base_goal[0]+1.4,"cart_y":base_goal[1],
                "reach_x":control["reach_x"],"reach_y":control["reach_y"],
                "lift":control["lift"]
            }
            for i,(key,goal) in enumerate(target.items()):
                q,v=ids[key]
                kp,kd,limit=((120,45,230) if key.startswith("cart") else
                            (95,16,88) if key.startswith("reach") else (115,13,80))
                ctrl=float(np.clip(kp*(goal-data.qpos[q])-kd*data.qvel[v],-limit,limit))
                data.ctrl[i]=ctrl
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("PHYSICS_DIVERGENCE_FAIL_CLOSED")
            if step%20==0:
                p=base_pos()
                row={"time_s":round(float(data.time),3),"state":phase,"object":current or "",
                  "base_x":round(p[0],4),"base_y":round(p[1],4),
                  "tool_z":round(float(data.site_xpos[site,2]),4),
                  "completed":len(accepted),"skipped":len(skipped),"failed":len(failed),
                  "active_contact_constraint":int(active_weld is not None)}
                log.append(row)
                if bridge.node:
                    bridge.emit(bridge.state_pub,dict(session_id=bridge.session_id,**row))
                    bridge.emit(bridge.phase_pub,{"session_id":bridge.session_id,
                          "state":phase,"time_s":row["time_s"],"object":current})
            if renderer is not None and data.time>=frame_due:
                try:
                    renderer.update_scene(data,camera=("pick_close" if phase in ("LOWER_TOOL","LIFT_VERIFICATION") else "bin_close" if phase in ("DROP_REINSPECT",) else "overview"))
                    arr=renderer.render()
                    img=Image.fromarray(arr).convert("RGB")
                    draw=ImageDraw.Draw(img)
                    draw.rectangle((0,0,848,77),fill=(15,26,39))
                    draw.text((14,7),"PAI | Physical AI 2026 | Litter pickup, ROS2 + MuJoCo",font=font,fill="white")
                    draw.text((14,32),f"STATE: {phase} | item: {current or 'none'} | collected {len(accepted)}/3",font=font,fill=(146,223,191))
                    draw.rectangle((0,456,848,480),fill=(17,24,35))
                    draw.text((12,460),"CONTACT-GATED SUCTION | Simulated appearance sensor | Holonomic drive proxy",fill="white")
                    video_writer.append_data(np.asarray(img))
                    frame_count+=1;frame_due+=1/args.video_fps
                except Exception as e:
                    print("C81_RENDER_ERROR",repr(e),flush=True)
                    raise
            if viewer:
                viewer.sync()
                if args.pace:
                    left=float(data.time)-(time.monotonic()-start_wall)
                    if left>.005:time.sleep(min(left,.03))
        result={
          "version":VERSION,
          "name":"大型商業施設の散乱ごみ自律回収ロボットの開発",
          "subtitle":"ROS 2とMuJoCoによるごみ認識・把持・回収・清掃完了判定",
          "limits":["MuJoCo-only, no real hardware","virtual appearance+geometry sensor, not real camera",
                    "holonomic cart drive proxy, not motorized wheels","3-axis telescopic arm, not 6DoF",
                    "contact-activated suction weld, not finger-friction grasp",
                    "simulated dry trash, not dirty waste or full indoor SLAM"],
          "session_id":bridge.session_id,"model_sha256":hashlib.sha256(xml_scene().encode()).hexdigest(),
          "ai_policy_sha256":SHA,
          "ai_over_external_ros":bool(bridge.use_ros),"ai_replies":bridge.received,"rejected_ros_messages":bridge.rejected,
          "collected":accepted,"skipped":skipped,"failed":failed,
          "predictions":labels,
          "frames":frame_count,"sim_time_s":round(float(data.time),3),
          "physics_e2e_pass":len(accepted)==3 and len(skipped)==1 and len(failed)==0 and phase=="DONE",
          "contact_gate":"MuJoCo geom-to-geom contact only",
          "object_position_teleport":"NONE (initial positions are in MJCF; dynamic object joints never set)"
        }
        with (args.out/"trace.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(log[0]))
            writer.writeheader();writer.writerows(log)
        (args.out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("C81_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
        if not result["physics_e2e_pass"]:raise AssertionError("C81_REAL_PHYSICS_PICK_NOT_YET_PASS")
        print("C81_ROS_MUJOCO_PHYSICAL_PICK_PLACE_PASS",flush=True)
        return result
    finally:
        if video_writer:video_writer.close()
        if renderer:renderer.close()
        if viewer:viewer.close()
        bridge.close()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("/tmp/pai_c81_result"))
    p.add_argument("--duration",type=float,default=135)
    p.add_argument("--no-ros",action="store_true")
    p.add_argument("--video",action="store_true")
    p.add_argument("--video-fps",type=int,default=3)
    p.add_argument("--viewer",action="store_true")
    p.add_argument("--pace",action="store_true")
    a=p.parse_args()
    do_run(a)
if __name__=="__main__":main()