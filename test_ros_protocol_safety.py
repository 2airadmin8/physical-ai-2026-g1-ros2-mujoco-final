"""Unit tests for ROS command replay, session isolation, payload and fail-closed."""
import json, time, types, unittest
import numpy as np
from pai_g1_closed_loop_ros_external import ExternalRosAI, SPEEDS, trained_policy
from pai_g1_ros_ai_policy import AIController
class Message:
    def __init__(self, data=""): self.data=data
class Publisher:
    def __init__(self,callback=None): self.callback=callback; self.messages=[]
    def publish(self,msg):
        self.messages.append(json.loads(msg.data))
        if self.callback: self.callback(msg)
class FakeNode:
    def __init__(self,on_observation=None):
        self.observation=Publisher(on_observation); self.command=Publisher()
    def create_publisher(self, msgtype, topic, qos):
        return self.observation if "sensor_observation" in topic else self.command
    def create_subscription(self, msgtype,topic,callback,qos):
        self.receiver=callback
        return callback
class FakeRos:
    String=Message
    def __init__(self, callback=None): self.node=FakeNode(callback); self.rclpy=types.SimpleNamespace(spin_once=lambda *a,**k:None)
class TestCommands(unittest.TestCase):
    def setUp(self):
        self.hash=trained_policy()["sha256"]
        self.ros=FakeRos()
        self.bridge=ExternalRosAI(self.ros,self.hash,timeout_s=0.08)
        self.bridge.awaiting_seq=1
    def send(self,session=None,seq=1,decision="STOP",sha=None,phase="FORKLIFT_PRESENT",speed=0.0):
        payload={"session_id":session if session is not None else self.bridge.session_id,
                 "seq":seq,"decision":decision,"risk":.993,
                 "requested_speed_m_s":speed,"policy_sha256":sha or self.hash,"phase":phase}
        self.bridge._on_command(Message(json.dumps(payload)))
        return payload
    def test_session_is_random_and_unique(self):
        other=ExternalRosAI(FakeRos(),self.hash)
        self.assertEqual(len(self.bridge.session_id),32)
        self.assertNotEqual(self.bridge.session_id,other.session_id)
    def test_reject_other_session(self):
        self.send(session="0"*32)
        self.assertFalse(self.bridge.pending)
        self.assertEqual(self.bridge.rejected_messages,1)
    def test_reject_other_sequence(self):
        self.send(seq=2)
        self.assertFalse(self.bridge.pending)
    def test_reject_duplicate(self):
        self.send()
        self.send()
        self.assertEqual(self.bridge.rejected_messages,1)
        self.assertEqual(len(self.bridge.pending),1)
    def test_reject_malformed(self):
        self.bridge._on_command(Message("{invalid"))
        self.assertFalse(self.bridge.pending)
        self.assertEqual(self.bridge.rejected_messages,1)
    def test_ai_command_roundtrip(self):
        def reply(obs):
            inp=json.loads(obs.data)
            response={"session_id":inp["session_id"],"seq":inp["seq"],"decision":"STOP",
                      "risk":0.993,"requested_speed_m_s":0.0,
                      "policy_sha256":self.hash,"phase":inp["phase"]}
            self.bridge._on_command(Message(json.dumps(response)))
        self.ros.node.observation.callback=reply
        decision,risk=self.bridge.infer(np.asarray([.15,7,15,1,1,1]),"FORKLIFT_PRESENT",9.55,20.)
        self.assertEqual(decision,"STOP")
        self.assertAlmostEqual(risk,.993)
        self.assertEqual(self.bridge.received,1)
        self.assertEqual(self.ros.node.observation.messages[0]["session_id"],self.bridge.session_id)
    def test_bad_sha_fails_closed(self):
        def reply(obs):
            inp=json.loads(obs.data)
            self.send(session=inp["session_id"],seq=inp["seq"],sha="f"*64)
        self.ros.node.observation.callback=reply
        with self.assertRaisesRegex(RuntimeError,"policy SHA mismatch"):
            self.bridge.infer(np.ones(6),"FORKLIFT_PRESENT",9.55,20.)
    def test_bad_phase_fails_closed(self):
        def reply(obs):
            inp=json.loads(obs.data)
            self.send(session=inp["session_id"],seq=inp["seq"],phase="WRONG")
        self.ros.node.observation.callback=reply
        with self.assertRaisesRegex(RuntimeError,"session/phase mismatch"):
            self.bridge.infer(np.ones(6),"FORKLIFT_PRESENT",9.55,20.)
    def test_bad_speed_fails_closed(self):
        def reply(obs):
            inp=json.loads(obs.data)
            self.send(session=inp["session_id"],seq=inp["seq"],speed=0.35)
        self.ros.node.observation.callback=reply
        with self.assertRaisesRegex(RuntimeError,"decision/speed mismatch"):
            self.bridge.infer(np.ones(6),"FORKLIFT_PRESENT",9.55,20.)
    def test_missing_policy_timeout(self):
        with self.assertRaisesRegex(TimeoutError,"EXTERNAL_ROS_AI_TIMEOUT"):
            self.bridge.infer(np.ones(6),"FORKLIFT_PRESENT",9.55,20.)
        self.assertIsNone(self.bridge.awaiting_seq)
if __name__=="__main__":
    unittest.main(verbosity=2)