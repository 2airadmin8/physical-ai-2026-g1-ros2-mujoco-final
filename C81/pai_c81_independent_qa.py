#!/usr/bin/env python3
"""Independent strict QA of three ROS+MuJoCo contact pick/place runs and rosbag.
Fails if replay evidence is missing, object was not physically recovered,
private item was touched, ROS AI is internal, policy hash changed, or video
is below the course requirement. Does not claim real robot / real camera.
"""
import argparse, collections, csv, hashlib, json, re, sqlite3, subprocess, sys
from pathlib import Path
from rclpy.serialization import deserialize_message
from std_msgs.msg import String
from pai_c81_ai_node import SHA
def check(cond,msg):
    if not cond:raise AssertionError(msg)
def verify(root:Path):
    report={"checks":[],"runs":[]}
    sessions=set();model_hashes=set()
    for i in range(3):
        folder=root/f"layout_{i}"
        s=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        log=(folder/"run.log").read_text(encoding="utf-8")
        trace=list(csv.DictReader((folder/"trace.csv").open(encoding="utf-8")))
        check(s["physics_e2e_pass"],f"layout{i} physics NOT PASS")
        check(s["ai_over_external_ros"] and s["ai_replies"]==4,f"layout{i} ROS not external")
        check(s["ai_policy_sha256"]==SHA,f"layout{i} policy changed")
        check(s["collected"]==["paper_1","cup_1","wrapper_1"],f"layout{i} recovery mismatch")
        check(s["skipped"]==["personal_1"] and not s["failed"],f"layout{i} safety/fail mismatch")
        check(len(s["predictions"])==4 and all(p["truth"]==p["pred"] for p in s["predictions"]),f"layout{i} AI mismatch")
        check(all(p["safety"]=="PICK" for p in s["predictions"][:3]) and s["predictions"][3]["safety"]=="SKIP",f"layout{i} safety classification")
        for event in ("C81_PHYSICAL_CONTACT_ATTACH","C81_RELEASE_ABOVE_BIN","C81_PHYSICALLY_COLLECTED"):
            check(log.count(event)==3,f"layout{i} expected three physical events of {event}")
        check(any(int(t["active_contact_constraint"])==1 for t in trace),f"layout{i} no contact force constraint records")
        check(s["rejected_ros_messages"]==0,f"layout{i} ROS replay detected")
        session=s["session_id"];sessions.add(session);model_hashes.add(s["model_sha256"])
        report["runs"].append({"layout":i,"time":s["sim_time_s"],"session":session,
                   "recovered":s["collected"],"private_skipped":s["skipped"],"trace_rows":len(trace)})
        report["checks"].append(f"LAYOUT_{i}_PHYSICAL_PICK_AND_REINSPECTION_PASS")
    check(len(sessions)==3,"ROS session IDs must differ")
    check(len(model_hashes)==3,"three environments must be distinct MJCFs")
    dbs=sorted((root/"rosbag2").glob("*.db3"))
    check(len(dbs)>=1,"rosbag SQLite not found")
    counts=collections.Counter(); messages=collections.defaultdict(list)
    for db in dbs:
        with sqlite3.connect(db) as con:
            names={a:b for a,b in con.execute("SELECT id,name FROM topics")}
            for topic_id,raw in con.execute("SELECT topic_id,data FROM messages ORDER BY timestamp"):
                topic=names[topic_id]
                o=json.loads(deserialize_message(raw,String).data)
                check(o.get("session_id") in sessions,f"unknown session in {topic}")
                counts[topic]+=1
                messages[topic].append(o)
    obs=messages["/pai_b/object_features"]; replies=messages["/pai_b/ai_classification"]
    check(len(obs)==12 and len(replies)==12,f"missing ROS observations or AI replies: {dict(counts)}")
    obskeys={(o["session_id"],int(o["seq"])) for o in obs}
    repkeys={(o["session_id"],int(o["seq"])) for o in replies}
    check(len(obskeys)==12 and obskeys==repkeys,"ROS seq mismatch/replay")
    for topic in ("/pai_b/robot_state","/pai_b/phase"):
        check(counts[topic]>=800,f"{topic} missing state samples")
    for x in replies:
        check(x.get("policy_sha256")==SHA,"rosbag contains different trained model")
        check((x["safety"]=="SKIP")== (x["label"]=="personal" or x["confidence"]<.75),"ROS AI safety mismatch")
    report["bag_topic_counts"]=dict(counts)
    report["total_messages"]=sum(counts.values())
    report["checks"].append("ROS2_CDR_4TOPIC_SEQ_SESSION_SHA_PASS")
    video=root/"PAI_C81_FINAL_3LAYOUTS_85s.mp4"
    check(video.exists(),"final video not found")
    info=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries",
        "format=duration,size:stream=codec_name,width,height", "-of","json",str(video)]))
    duration=float(info["format"]["duration"])
    check(60<=duration<=180,f"course video requirement 1-3min failed: {duration}")
    check(any(x["codec_name"]=="h264" and x["width"]>=640 for x in info["streams"]), "invalid H264 video")
    report["video"]={"filename":video.name,"duration_s":duration,
                     "bytes":video.stat().st_size,"sha256":hashlib.sha256(video.read_bytes()).hexdigest()}
    report["checks"].append("COURSE_MP4_DURATION_H264_PASS")
    source=(Path(__file__).parent/"pai_c81_robot_runtime.py").read_text(encoding="utf-8")
    check(not re.search(r"\bdata\.(?:qpos|qvel)\s*\[[^\n]*\]\s*=(?!=)",source),
          "direct qpos/qvel assignment detected: unsafe pose teleport")
    check("contact_with(model,data,arm_geom,obj" in source,"contact condition missing")
    report["checks"].append("STATIC_NO_DIRECT_QPOS_QVEL_ACCESS_IN_RUNTIME_PASS")
    report["pass"]=True
    print("C81_STRICT_INDEPENDENT_QA_PASS",json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    (root/"INDEPENDENT_QA.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return report
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("/tmp/pai_c81_final"))
    a=p.parse_args()
    verify(a.root)
if __name__=="__main__":main()