#!/usr/bin/env python3
"""Strict read-only CDR rosbag audit for PAI v1.3 external ROS control."""
import argparse, json, sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

TOPICS=[
"/pai_final/sensor_observation",
"/pai_final/control_command",
"/pai_final/ai_decision",
"/pai_final/robot_feedback"
]
def audit(folder:Path, count_complete=False):
    dbs=sorted(folder.glob("*.db3"))
    assert len(dbs)>=1, "No bag DB files"
    counts=Counter()
    sessions=defaultdict(set)
    seqs=defaultdict(list)
    errors=[]
    decision_counts=Counter()
    for file in dbs:
        with sqlite3.connect(file) as db:
            names={i:n for i,n in db.execute("select id,name from topics")}
            for tid,raw in db.execute("select topic_id,data from messages order by timestamp"):
                topic=names[tid]
                if topic not in TOPICS:
                    continue
                msg=deserialize_message(raw,String)
                o=json.loads(msg.data)
                counts[topic]+=1
                sid=o.get("session_id")
                if not sid or len(sid)!=32:
                    errors.append(f"Missing/bad session_id topic={topic}")
                sessions[topic].add(sid)
                if "seq" in o:
                    seqs[topic].append(int(o["seq"]))
                if topic=="/pai_final/ai_decision":
                    decision_counts[o["decision"]]+=1
                if topic=="/pai_final/control_command":
                    assert o.get("policy_sha256")== "9a606472b1daae2b9c892656be1a47adf329f9b1ad609910cc409d526a14839a"
    keys=set.union(*(set(x) for x in sessions.values()))
    if len(keys)!=1: errors.append(f"Cross-session mismatch: {list(keys)}")
    for topic in TOPICS:
        if counts[topic]<2000:
            errors.append(f"Too few {topic}={counts[topic]}")
    obs=set(seqs["/pai_final/sensor_observation"])
    cmd=set(seqs["/pai_final/control_command"])
    if obs != cmd:
        errors.append(f"Observation/command seq mismatch missing_obs={sorted(cmd-obs)[:10]} missing_cmd={sorted(obs-cmd)[:10]}")
    for topic,keys2 in seqs.items():
        if len(set(keys2))!=len(keys2):
            errors.append(f"Duplicate command/observation seq in {topic}")
    if count_complete and dict(counts)!={TOPICS[0]:2220,TOPICS[1]:2220,TOPICS[2]:2160,TOPICS[3]:2160}:
        errors.append(f"Incomplete expected counters {dict(counts)}")
    report={
        "passed":not errors,"bag":str(folder),"topic_counts":dict(counts),
        "observed_session_ids":list(keys),
        "decision_counts":dict(decision_counts),
        "unique_observation_seqs":len(obs),
        "unique_command_seqs":len(cmd),
        "errors":errors
    }
    print("PAI_ROSBAG_CDR_AUDIT_"+("PASS" if not errors else "FAIL"))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--strict-count",action="store_true")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    result=audit(a.folder,a.strict_count)
    if a.out:
        a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not result["passed"]:raise SystemExit(1)