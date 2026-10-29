#!/usr/bin/env python3
"""Independent 2026-10-30 ROS cross-process + MuJoCo physics evidence gate.

Do not mistake this G1 rail-assisted simulation for physical-robot locomotion.
"""
import argparse
import json
import sqlite3
from pathlib import Path

def verify(run_dir: Path, rosbag_dir: Path) -> dict:
    summaries=[]
    for i in range(1,4):
        f=run_dir/f"ai_{i}.json"
        item=json.loads(f.read_text(encoding="utf-8"))
        assert item["ai_command_over_ros"] is True, f"run{i}: no ROS AI command path"
        assert item["physics_engine"]=="3.14.0", f"run{i}: wrong physics engine"
        assert item["stop_count"]==25 and item["slow_count"]==201 and item["go_count"]==494
        assert abs(item["end_x_m"]-12.8475)<0.0005
        assert "independent ROS ML policy" in item["control_path"]
        summaries.append(item)
    ab=json.loads((run_dir/"counterfactual.json").read_text(encoding="utf-8"))
    assert ab["causal_delta_m"]>1.5, ab
    assert abs(ab["stop_end_x"]-9.55603)<0.001, ab
    assert abs(ab["forced_go_end_x"]-11.12148)<0.001, ab
    dbs=sorted(rosbag_dir.glob("*.db3"))
    assert dbs, "Missing ROS 2 SQLite bag"
    counts={}
    for db in dbs:
        with sqlite3.connect(db) as cn:
            rows=cn.execute(
                "SELECT topics.name,COUNT(messages.id) "
                "FROM topics LEFT JOIN messages ON topics.id=messages.topic_id "
                "GROUP BY topics.name").fetchall()
            for topic,num in rows:
                counts[topic]=counts.get(topic,0)+num
    expected={
        "/pai_final/sensor_observation":2220,
        "/pai_final/control_command":2220,
        "/pai_final/ai_decision":2160,
        "/pai_final/robot_feedback":2160
    }
    for name, num in expected.items():
        assert counts.get(name)==num, (name,counts.get(name),num)
    result={
        "pass":True,
        "mode":"external ROS policy -> topic -> MuJoCo actuator -> mj_step",
        "simulator":"MuJoCo 3.14.0, Unitree G1 29DoF rail-assisted fixture",
        "three_end_x_m":[x["end_x_m"] for x in summaries],
        "policy_sha256":summaries[0]["risk_model_sha256"],
        "counterfactual_delta_m":ab["causal_delta_m"],
        "bag_counts":counts,
        "bag_total_messages":sum(counts.values()),
        "scope_warning":"simulation only; no unsupported biped walking and no hardware G1 control"
    }
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--bag-dir",type=Path,required=True)
    p.add_argument("--save",type=Path)
    args=p.parse_args()
    result=verify(args.run_dir,args.bag_dir)
    print("PAI_EXTERNAL_ROS_MUJOCO_VALIDATOR_PASS")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.save:
        args.save.parent.mkdir(parents=True,exist_ok=True)
        args.save.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")