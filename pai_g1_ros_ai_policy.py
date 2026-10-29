#!/usr/bin/env python3
"""Independent ROS 2 AI policy node, separate from the G1 MuJoCo actuator process.

ROS sensor_observation -> real logistic-regression inference -> control_command.
The MuJoCo process never computes AI risk in --external-ros-ai mode.
Synthetic examples and simulated hazards: not certified physical-robot safety.
"""
import json
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pai_g1_closed_loop_ros_external import trained_policy, ai_inference, SPEEDS

class AIController(Node):
    def __init__(self):
        super().__init__("pai_g1_ai_ros_policy")
        self.policy=trained_policy()
        self.cmd_pub=self.create_publisher(String, "/pai_final/control_command", 10)
        self.obs_sub=self.create_subscription(String, "/pai_final/sensor_observation", self.on_observation, 10)
        self.counts={"GO":0,"SLOW":0,"STOP":0}
        self.sent=0
        self.last_decision=None
        print("ROS_POLICY_READY",self.policy["sha256"], flush=True)

    def on_observation(self, msg):
        try:
            data=json.loads(msg.data)
            features=np.asarray(data["features"],dtype=np.float64)
            if features.shape!=(6,) or not np.isfinite(features).all():
                raise ValueError("Invalid sensor feature vector")
            decision, risk = ai_inference(self.policy, features)
            session_id=str(data["session_id"])
            if len(session_id)!=32 or any(c not in "0123456789abcdef" for c in session_id):
                raise ValueError("Invalid session_id")
            response={
                "seq": int(data["seq"]),
                "session_id": session_id,
                "decision": decision,
                "risk": float(risk),
                "requested_speed_m_s": SPEEDS[decision],
                "policy_sha256":self.policy["sha256"],
                "phase":str(data.get("phase","")),
            }
            out=String()
            out.data=json.dumps(response,sort_keys=True)
            self.cmd_pub.publish(out)
            self.counts[decision]+=1
            self.sent+=1
            if decision!=self.last_decision:
                print("ROS_POLICY_COMMAND","seq="+str(response["seq"]),
                      "decision="+decision,"risk="+format(risk,".4f"),
                      "phase="+response["phase"],flush=True)
                self.last_decision=decision
            if self.sent%100==0:
                print("ROS_POLICY_SENT",self.sent,self.counts,flush=True)
        except Exception as exc:
            print("ROS_POLICY_ERROR",type(exc).__name__,str(exc),flush=True)

def main():
    rclpy.init()
    node=AIController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("ROS_POLICY_FINAL",node.sent,node.counts,flush=True)
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()