param(
    [string]$Container = "pai-ros2-course-061",
    [string]$Root = "$env:USERPROFILE\PAI_Final_Assignment_2026",
    [switch]$SkipPackageInstall
)
$ErrorActionPreference = 'Stop'
$repo = Join-Path $Root 'unitree_mujoco'
$g1 = Join-Path $repo 'unitree_robots\g1'
$commit = '1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d'
if(-not (Test-Path $repo)) {
  & git clone --filter=blob:none https://github.com/unitreerobotics/unitree_mujoco.git $repo
  if($LASTEXITCODE -ne 0){throw "git clone failed"}
}
$head=(& git -C $repo rev-parse HEAD).Trim()
if($head -ne $commit){
  & git -C $repo checkout $commit
  if($LASTEXITCODE -ne 0){throw "Pinned model commit missing"}
  $head=(& git -C $repo rev-parse HEAD).Trim()
}
if($head -ne $commit){throw "Model commit mismatch $head"}
$scene=Join-Path $Root 'scene_factory_demo.xml'
if(-not (Test-Path $scene)){throw "Missing scene_factory_demo.xml in release"}
Copy-Item $scene (Join-Path $g1 'scene_factory_demo.xml') -Force
$state=(& docker inspect $Container --format '{{.State.Running}}' 2>$null)
if($state -ne 'true'){throw "Expected running ROS course container $Container"}
& docker exec $Container bash -lc 'mkdir -p /home/ubuntu/pai_closed_loop/g1'
if($LASTEXITCODE -ne 0){throw 'Target dir failed'}
& docker cp "$g1\." "$($Container):/home/ubuntu/pai_closed_loop/g1"
if($LASTEXITCODE -ne 0){throw "G1 model transfer failed"}
foreach($n in @('pai_g1_closed_loop_ros_external.py','pai_g1_ros_ai_policy.py','test_ros_protocol_safety.py','audit_rosbag_cdr.py')){
  $p=Join-Path $Root $n
  if(-not (Test-Path $p)){throw "Missing script $n"}
  & docker cp $p "$($Container):/tmp/$n"
  if($LASTEXITCODE -ne 0){throw "docker cp failed: $n"}
}
if(-not $SkipPackageInstall){
  & docker exec -u ubuntu $Container bash -lc 'python3 -m pip install --user --no-cache-dir mujoco==3.14.0 imageio==2.37.0 imageio-ffmpeg==0.6.0'
  if($LASTEXITCODE -ne 0){throw 'MuJoCo package install failed'}
}
& docker exec -u ubuntu $Container bash -lc 'source /opt/ros/humble/setup.bash; python3 -m py_compile /tmp/pai_g1_closed_loop_ros_external.py /tmp/pai_g1_ros_ai_policy.py; cd /tmp && python3 -m unittest test_ros_protocol_safety -q'
if($LASTEXITCODE -ne 0){throw 'Protocol tests failed'}
Write-Output "PAI_FRESH_BOOTSTRAP_PASS container=$Container model=$commit"