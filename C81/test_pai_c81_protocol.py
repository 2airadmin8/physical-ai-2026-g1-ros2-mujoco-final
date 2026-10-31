#!/usr/bin/env python3
"""Independent C81 protocol/physics/model unit tests, no business containers."""
import json,unittest
from types import SimpleNamespace
import numpy as np
import mujoco
from pai_c81_ai_node import CLASSES,PROTOTYPES,classify,create_model,SHA
from pai_c81_robot_runtime import Bridge,contact_with,set_weld
from pai_c81_mujoco import xml_scene
class Tests(unittest.TestCase):
    def test_model_version_and_dynamics(self):
        m=mujoco.MjModel.from_xml_string(xml_scene())
        self.assertEqual(m.nu,5)
        self.assertEqual(m.neq,4)
        d=mujoco.MjData(m)
        self.assertFalse(any(d.eq_active))
        mujoco.mj_forward(m,d)
        x=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"paper_1")
        self.assertTrue(np.isfinite(d.xpos[x]).all())
    def test_policy_four_labels(self):
        for i,p in enumerate(PROTOTYPES):
            label,prob=classify(p)
            self.assertEqual(label,CLASSES[i])
            self.assertGreater(prob,.75)
    def test_policy_noise_tolerance(self):
        rng=np.random.default_rng(99)
        for i,p in enumerate(PROTOTYPES):
            for _ in range(10):
                value=p+rng.normal(0,[.012,.012,.012,.001,.001,.002])
                self.assertEqual(classify(value)[0],CLASSES[i])
    def test_sha_is_fixed(self):
        self.assertEqual(create_model()[2],SHA)
        self.assertEqual(len(SHA),64)
    def test_separate_ros_message_replay_rejected(self):
        bridge=Bridge(False)
        bridge.awaiting=1
        payload={"session_id":bridge.session_id,"seq":1,"label":"paper",
                 "confidence":.9,"policy_sha256":SHA,"safety":"PICK"}
        bridge.receive(SimpleNamespace(data=json.dumps({**payload,"session_id":"0"*32})))
        bridge.receive(SimpleNamespace(data=json.dumps({**payload,"seq":2})))
        bridge.receive(SimpleNamespace(data=json.dumps(payload)))
        bridge.receive(SimpleNamespace(data=json.dumps(payload)))
        self.assertEqual(bridge.rejected,3)
        self.assertEqual(len(bridge.pending),1)
        self.assertEqual(bridge.pending[1]["label"],"paper")
    def test_contact_guard_no_grab_at_initial_distance(self):
        m=mujoco.MjModel.from_xml_string(xml_scene())
        d=mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        suction=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"suction")
        paper=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"g_paper_1")
        self.assertFalse(contact_with(m,d,suction,paper))
    def test_no_real_camera_or_hardware_claim(self):
        source=__import__("inspect").getsource(__import__("pai_c81_robot_runtime"))
        self.assertIn("simulated color+shape sensor",source)
        self.assertIn("contact_with(model,data,arm_geom",source)
if __name__=="__main__":
    unittest.main(verbosity=2)