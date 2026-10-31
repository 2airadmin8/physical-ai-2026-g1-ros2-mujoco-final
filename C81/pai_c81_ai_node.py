#!/usr/bin/env python3
"""Independent ROS2 educational AI classifier for MuJoCo geometric-color sensor.
No real camera; 4-class synthetic appearance+geometry classification via an
actual multiclass softmax trained on synthetic labeled examples.
"""
import hashlib,json,os
import numpy as np
CLASSES=("paper","cup","wrapper","personal")
PROTOTYPES=np.array([
 [.85,.81,.65, .055,.043,.025],
 [.96,.47,.15, .057,.057,.075],
 [.19,.66,.88, .075,.050,.025],
 [.08,.08,.10, .080,.045,.022],
],dtype=float)
FEATURE_SCALE=np.array([1,1,1,.1,.1,.1],dtype=float)
def create_model():
    seed=41026
    g=np.random.default_rng(seed)
    xs=[];ys=[]
    for i,p in enumerate(PROTOTYPES):
        for _ in range(48):
            noise=g.normal(0, [.027,.027,.027,.002,.002,.004])
            xs.append((p+noise)/FEATURE_SCALE)
            ys.append(i)
    X=np.asarray(xs); Y=np.eye(4)[ys]
    W=np.zeros((6,4)); b=np.zeros(4)
    for i in range(1600):
        z=np.clip(X@W+b,-35,35); z-=z.max(axis=1,keepdims=True)
        e=np.exp(z);prob=e/e.sum(axis=1,keepdims=True)
        W-=.22*X.T@(prob-Y)/len(X)
        b-=.22*np.mean(prob-Y,axis=0)
    W=np.round(W,7);b=np.round(b,7)
    sha=hashlib.sha256(W.astype('<f8').tobytes()+b.astype('<f8').tobytes()).hexdigest()
    return W,b,sha
W,BIAS,SHA=create_model()
def classify(features):
    x=np.asarray(features,dtype=float)
    if x.shape!=(6,) or not np.isfinite(x).all():raise ValueError("six valid numeric features required")
    z=np.clip((x/FEATURE_SCALE)@W+BIAS,-35,35)
    z=z-z.max();p=np.exp(z);p/=p.sum()
    k=int(p.argmax())
    return CLASSES[k],float(p[k])
def main():
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node=rclpy.create_node("pai_c81_independent_ai_policy")
    pub=node.create_publisher(String,"/pai_b/ai_classification",10)
    seen=set()
    def cb(msg):
        try:
            x=json.loads(msg.data)
            sid=x["session_id"]
            seq=int(x["seq"])
            if len(sid)!=32 or seq<1:return
            if (sid,seq) in seen:return
            seen.add((sid,seq))
            label,confidence=classify(x["features"])
            answer={"session_id":sid,"seq":seq,"label":label,
                    "confidence":round(confidence,6),"policy_sha256":SHA,
                    "safety":"SKIP" if label=="personal" or confidence<.75 else "PICK"}
            m=String();m.data=json.dumps(answer,separators=(",",":"))
            pub.publish(m)
            print("AI_CLASSIFY",seq,label,round(confidence,4),answer["safety"],flush=True)
        except Exception as e:print("AI_REJECT",repr(e),flush=True)
    node.create_subscription(String,"/pai_b/object_features",cb,10)
    print("PAI_C81_AI_POLICY_READY",SHA,flush=True)
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()
if __name__=="__main__":main()