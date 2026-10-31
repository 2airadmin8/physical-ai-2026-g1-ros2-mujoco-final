#!/usr/bin/env python3
"""PAI C81: honest MuJoCo contact-activated vacuum pick/place (simulation only).
Robot is an omnidirectional planar-drive PROXY and 3-axis gantry arm, NOT a
physical AMR, full 6-DoF arm, autonomous biped, or real-camera RGB-D perception.
Physics is advanced by applied actuator forces + mj_step only. Free trash bodies
move only from contacts, gravity, and dynamically activated suction welds.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mujoco

VERSION="0.1.0"
NAMES=["paper","cup","wrapper","personal"]
KINDS={
"paper": (".85 .81 .65 1", ".055 .043 .025", 0.030),
"cup": (".96 .47 .15 1", ".057 .057 .075", 0.065),
"wrapper": (".19 .66 .88 1", ".075 .050 .025", 0.025),
"personal": (".08 .08 .10 1", ".080 .045 .022", 0.150),
}
OBJECTS=[
 ("paper_1","paper",(-.1,.28)),
 ("cup_1","cup",(.75,-.10)),
 ("wrapper_1","wrapper",(1.43,.40)),
 ("personal_1","personal",(.34,-.62)),
]
# Three independent INITIAL MuJoCo layouts; no object pose teleport in the loop.
LAYOUT=int(os.environ.get("PAI_C81_LAYOUT","0"))
if LAYOUT==1:
    OBJECTS=[("paper_1","paper",(-.25,-.25)),
             ("cup_1","cup",(.82,.34)),
             ("wrapper_1","wrapper",(1.28,-.08)),
             ("personal_1","personal",(.35,-.70))]
elif LAYOUT==2:
    OBJECTS=[("paper_1","paper",(.10,.52)),
             ("cup_1","cup",(.66,-.26)),
             ("wrapper_1","wrapper",(1.56,.35)),
             ("personal_1","personal",(-.28,-.65))]
elif LAYOUT!=0:
    raise ValueError("Only layouts 0,1,2 are supported")
BIN=(2.5,-1.52)
def xml_scene():
    body=[]
    weld=[]
    for name,kind,(x,y) in OBJECTS:
        color,size,mass=KINDS[kind]
        sz=np.array([float(z) for z in size.split()])
        z=sz[2]+.005
        geom="cylinder" if kind=="cup" else "box"
        dim=f"{sz[0]} {sz[2]}" if geom=="cylinder" else size
        body.append(f'''<body name="{name}" pos="{x} {y} {z}">
        <freejoint name="free_{name}"/>
        <geom name="g_{name}" type="{geom}" size="{dim}"
              rgba="{color}" mass="{mass}" friction="1.4 .01 .003"
              solref=".012 1" contype="1" conaffinity="1"/>
        </body>''')
        weld.append(f'<weld name="vacuum_{name}" body1="tool" body2="{name}" active="false" torquescale="1.0"/>')
    return f"""<mujoco model="PAI_C81_litter_pick_and_recover">
<option timestep=".005" gravity="0 0 -9.81" integrator="implicitfast"/>
<compiler angle="radian"/>
<default>
 <joint limited="true" damping="5"/>
 <geom contype="1" conaffinity="1" friction="1.2 .015 .005" solref=".012 1"/>
</default>
<visual><global offwidth="960" offheight="540"/><headlight ambient=".45 .45 .45" diffuse=".6 .6 .6" specular=".1 .1 .1"/></visual>
<worldbody>
 <light name="area_light" pos="-2 -2 5" dir="0 0 -1" diffuse="1 1 1"/>
 <geom name="floor" type="plane" size="5 5 .1" rgba=".75 .78 .81 1" friction="1.1 .01 .001"/>
 <geom name="lane_left" type="box" pos="-.2 1.17 .045" size="3.1 .035 .045" rgba=".30 .38 .42 1" contype="0" conaffinity="0"/>
 <geom name="lane_right" type="box" pos="-.2 -1.08 .045" size="3.1 .035 .045" rgba=".30 .38 .42 1" contype="0" conaffinity="0"/>
 <geom name="foodcourt_border" type="box" pos="-1.2 1.72 .30" size=".85 .36 .30" rgba=".61 .46 .33 1"/>
 <geom name="chair_1" type="box" pos="-.95 1.18 .18" size=".19 .16 .18" rgba=".66 .51 .37 1"/>
 <geom name="chair_2" type="box" pos="1.65 1.36 .18" size=".19 .16 .18" rgba=".66 .51 .37 1"/>
 <geom name="plant" type="cylinder" pos="2.95 1.28 .25" size=".20 .25" rgba=".18 .56 .40 1"/>
 <geom name="bin_floor" type="box" pos="{BIN[0]} {BIN[1]} .05" size=".29 .29 .05" rgba=".10 .51 .27 1"/>
 <geom name="bin_w1" type="box" pos="{BIN[0]} {BIN[1]-.285} .26" size=".30 .018 .22" rgba=".08 .42 .23 1"/>
 <geom name="bin_w2" type="box" pos="{BIN[0]} {BIN[1]+.285} .26" size=".30 .018 .22" rgba=".08 .42 .23 1"/>
 <geom name="bin_w3" type="box" pos="{BIN[0]-.285} {BIN[1]} .26" size=".018 .30 .22" rgba=".08 .42 .23 1"/>
 <geom name="bin_w4" type="box" pos="{BIN[0]+.285} {BIN[1]} .26" size=".018 .30 .22" rgba=".08 .42 .23 1"/>
 <site name="bin_target" pos="{BIN[0]} {BIN[1]} .62" size=".06" rgba="0 1 0 1"/>
 <body name="cart" pos="-1.4 0 .16">
  <joint name="cart_x" type="slide" axis="1 0 0" limited="true" range="-2.5 4.5" damping="12"/>
  <joint name="cart_y" type="slide" axis="0 1 0" limited="true" range="-2.0 1.1" damping="12"/>
  <geom name="cart_frame" type="box" size=".22 .22 .13" rgba=".16 .34 .68 1" mass="17" contype="0" conaffinity="0"/>
  <geom name="wheel_l" type="cylinder" pos="0 .22 -.045" euler="1.5708 0 0" size=".075 .036" rgba=".14 .16 .18 1" contype="0" conaffinity="0"/>
  <geom name="wheel_r" type="cylinder" pos="0 -.22 -.045" euler="1.5708 0 0" size=".075 .036" rgba=".14 .16 .18 1" contype="0" conaffinity="0"/>
  <geom name="mast" type="capsule" fromto="0 0 .10 0 0 .77" size=".04" rgba=".77 .84 .86 1" contype="0" conaffinity="0"/>
  <camera name="pick_close" pos=".18 -1.00 1.35" euler=".67 0 0" fovy="55"/>
  <body name="arm_xbody" pos="0 0 .30">
   <joint name="reach_x" axis="1 0 0" type="slide" range="-.08 .64" damping="5"/>
   <geom type="box" size=".12 .08 .06" mass=".75" rgba=".88 .78 .28 1" contype="0" conaffinity="0"/>
   <body name="arm_ybody" pos="0 0 .03">
    <joint name="reach_y" axis="0 1 0" type="slide" range="-.12 .12" damping="5"/>
    <geom type="box" size=".09 .06 .05" mass=".45" rgba=".88 .78 .28 1" contype="0" conaffinity="0"/>
    <body name="tool" pos="0 0 .53">
     <joint name="lift" axis="0 0 1" type="slide" range="-.73 0" damping="4"/>
     <geom name="tool_pole" type="cylinder" pos="0 0 -.12" size=".028 .17" mass=".27" rgba=".92 .91 .86 1" contype="0" conaffinity="0"/>
     <geom name="suction" type="sphere" pos="0 0 -.31" size=".047" rgba=".96 .24 .33 1" mass=".10" solref=".007 1"/>
     <site name="suction_site" pos="0 0 -.31" size=".04" rgba="1 .3 .3 .7"/>
    </body>
   </body>
  </body>
 </body>
 {''.join(body)}
 <camera name="overview" pos="0 -3.85 5.0" euler=".64 0 0" fovy="45"/>
 <camera name="bin_close" pos="2.6 -2.9 2.35" euler=".67 0 0" fovy="55"/>
 <camera name="topdown" pos=".15 -.2 8.5" euler="0 0 0" fovy="46"/>
</worldbody>
<actuator>
<motor name="x_force" joint="cart_x" gear="1" ctrllimited="true" ctrlrange="-250 250"/>
<motor name="y_force" joint="cart_y" gear="1" ctrllimited="true" ctrlrange="-250 250"/>
<motor name="reach_force" joint="reach_x" gear="1" ctrllimited="true" ctrlrange="-90 90"/>
<motor name="lateral_force" joint="reach_y" gear="1" ctrllimited="true" ctrlrange="-90 90"/>
<motor name="lift_force" joint="lift" gear="1" ctrllimited="true" ctrlrange="-85 85"/>
</actuator>
<equality>{''.join(weld)}</equality>
</mujoco>"""

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("/tmp/pai_c81_result"))
    p.add_argument("--duration",type=float,default=110)
    p.add_argument("--no-render",action="store_true")
    p.add_argument("--no-ros",action="store_true")
    p.add_argument("--video-fps",type=int,default=4)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    xml=xml_scene()
    (args.out/"model.xml").write_text(xml,encoding="utf-8")
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    print("MODEL_LOADED",model.nq,model.nu,model.neq,model.opt.timestep,flush=True)
    for name,kind,xy in OBJECTS:
        e=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,"vacuum_"+name)
        print("EQ",name,e,model.eq_data[e,:7],flush=True)
    if args.dry_run:return
    print("IMPLEMENTATION_PENDING")
if __name__=="__main__":
    main()