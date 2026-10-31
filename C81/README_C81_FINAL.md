# Physical AI 2026｜C81 最終課題 v0.2.0

## 正式研究テーマ
大型商業施設の散乱ごみ自律回収ロボットの開発
副題：― ROS 2とMuJoCoによるごみ認識・把持・回収・清掃完了判定 ―

## このプロトタイプが実装したこと
- MuJoCo 3.14の通路、フードコート風什器、収集箱、自由剛体の紙片・カップ・包装・私物。
- 水平移動台車のXY駆動近似（力制御）＋3軸直動のピックアップ工具。
- AIの入力はMuJoCo内の外観色と物体寸法を使う仮想センサ。実カメラ画像認識ではない。
- 独立ROS2 AIノード：192件の合成例で学習した4クラスsoftmaxが、paper/cup/wrapper/personalを分類。
- ROSトピック observation→AI classification→受信セッション/seq/policy SHA検証→ロボット動作。
- 現実のMuJoCoジオメトリ同士が接触した場合のみ、真空吸着を模擬するweld拘束をON。アクチュエータ力とmj_stepで持ち上げ、移送、解除して重力で箱に落とし、再検査する。
- 私物は拾わず、安全スキップ。AI応答が無ければ一歩も進めないfail-closed。

## 正確な技術境界（必ず開示）
1. 実店舗、イオンモールとの提携・現場試験・許諾、実際の清掃実績はない。
2. 視覚認識は現実のRGB-Dカメラ映像ではなく、仮想色/寸法センサの学習済み分類である。
3. 移動ベースは平面XYに力を与える全方向走行の簡略モデル。実際の車輪駆動、真のSLAM、段差対応ではない。
4. マニピュレータは3軸直動機構であり、6軸腕や実物の指摩擦グリッパーではない。真空吸着は接触で有効化する物理拘束としてモデル化。
5. 液体、粘着した厨余、ガラス、針、人がいる営業時間中の仕事は対象外。
6. 企業の作業負担や人員削減・ROIは未実地調査であり、実証済み事実ではない。

## ソースコード
pai_c81_mujoco.py：3レイアウトのMuJoCo XMLと物体/回収箱
pai_c81_ai_node.py：独立ROS2 softmax AIモデル
pai_c81_robot_runtime.py：操作状態機械、接触条件、ピック/運搬/投箱/再検査
run_pai_c81_final3.sh：3実験＋ROS rosbag2＋H264 85秒動画
pai_c81_independent_qa.py：ログ/rosbag/3実験/動画の独立検証
test_pai_c81_protocol.py：7本の独立ユニットテスト
C81_final_20261030：CSV/JSON/SQLite/MPEG4/QAの実証証拠

## 実証結果（2026-10-30）
layout 0：紙片、カップ、包装 3/3 物理接触→持上→投箱完了。私物1/1 SKIP。
layout 1：同じ3種の異なる初期配置でも3/3回収。私物1/1 SKIP。
layout 2：さらに異なる配置でも3/3回収。私物1/1 SKIP。
合計9/9件回収・3/3件私物スキップは限定したシミュレーション条件内の結果で、実環境未知物の成功率ではない。

独立ROS2応答：3実験、全12セッション内seqに対応する特徴入力12件、分類返信12件。
ロボット状態851件、フェーズ851件、合計ROS bag2の1726メッセージをCDRデコード・モデルSHA・セッション一致・全seq一致で検証。
動画：H264、848×480、85.167秒、3レイアウト。最終サマリ/監査JSONで確認。
ユニットテスト7/7 PASS、ROS AI不在の隔離ドメインで4秒タイムアウト→物理計算ステップゼロのfail-closed PASS。
初回の底盤駆動範囲不足は原因特定後に修正。修正前の失敗を成功データに混ぜていない。

## AA8-N-0061上の再現
授業用Dockerコンテナ pai-ros2-course-061 を起動し、上記py/shをコンテナの /tmp に転送。
コマンド1（別独立AIプロセス）：
docker exec -d -u ubuntu pai-ros2-course-061 bash -lc 'source /opt/ros/humble/setup.bash; cd /tmp; python3 pai_c81_ai_node.py >/tmp/pai_c81_policy.log 2>&1'
コマンド2（単回）：
docker exec -u ubuntu -e DISPLAY=:0 pai-ros2-course-061 bash -lc 'source /opt/ros/humble/setup.bash; cd /tmp; python3 pai_c81_robot_runtime.py --out /tmp/demo_new --video --video-fps 6'
コマンド3（3配置、記録先が空であることを確認）：
docker exec -u ubuntu pai-ros2-course-061 bash /tmp/run_pai_c81_final3.sh
コマンド4（独立QA）：
docker exec -u ubuntu pai-ros2-course-061 bash -lc 'source /opt/ros/humble/setup.bash; cd /tmp; python3 pai_c81_independent_qa.py --root /tmp/pai_c81_final && python3 -m unittest test_pai_c81_protocol -v'

## 録画と提出
完成した85.167秒のMP4をWindowsのChromeで全画面表示し、本人が画面録画できる。
OmniCampusに出すPDF名/YouTubeタイトルは正式アカウント名またはIDが必要。本名/所属/メール等の個人情報を入れない。
YouTubeアップロードとOmniCampus最終受領が確認できるまで提出完了ではない。