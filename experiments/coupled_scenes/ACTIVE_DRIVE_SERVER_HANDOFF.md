# 主动驱动代码与服务器资产交接

本提交提供本地已使用的主动驱动实现及 CPU 测试；不声明 Windows 编排脚本已完成 Linux 适配。
不要覆盖服务器现有脏工作树，不自动启动仿真。以包含本文件的固定提交建立干净 detached worktree。

## 已发布资产

根目录 `/data/jiachen/assets/active_drive/active_drive_20260912_v1`，先读其中 README.md、manifest.json。
manifest SHA256：`954f5f732aca53b447254a358317c1ae6f3e2f66bc0377883c2f758b56e3249f`。
三套外观和四组碰撞/填水已在服务器 Blender 5.0.1 CPU 模式校验，详见 acceptance_final.json。
应用代码不在资产包内；原始 blend 不得重新保存来覆盖来源哈希。

## 当前待接入任务：两倍速搅拌

已有人工请求 `/data/jiachen/jobs/blocked/20260912_001_stirring_2x_handoff.json`。
其“代码和资产未到齐”描述是投递时状态，不能修改不可变原件；现在资产已到，代码由本提交交接。
接入后请以新 ID 建立 pipeline_job，不原地重投旧请求。

仿真入口：`experiments/coupled_scenes/run_active_pour_probe.py`。
通过服务器 `isaac_docker_python.py` 调用，脚本参数：

```text
--assets /data/jiachen/assets/active_drive/active_drive_20260912_v1/simulation_assets/02_stirring
--output <新任务输出目录>/simulation
--restore-vorticity-after-settling
--relaxed-restored-readiness
--recycle-stirring-prewarm
--stirring-speed 3.6
--seconds 10
```

不要原样执行含占位符的示例。使用4mm、694674粒子、720Hz/64迭代；预热涡量.02，
就绪后平滑恢复10并通过原有恢复门槛。10秒是预热后的动作窗口：0–2秒不动，
2–3秒起转，3–7秒满速，7–8秒停止，8–10秒余流。不要改为静水官方Water涡量0配方。
局部回收只在预热阶段，最多64个碗外下降粒子；动作中不删飞溅、不掩盖碗底漏水。
旧任务因 Windows 写报告 WinError5 在动作2.93秒退出，当前代码已有有界重试。
旧NPZ不是求解器检查点；此次应从初始资产重跑。

## Linux 接入待办（服务器 agent）

1. 使用固定提交的干净 worktree 与现有单卡运行包装器。原生仿真入口可单独调用，
   但服务器 Isaac/PhysX 兼容性尚未实跑验证；先检查 imports/参数/资产，再进行授权仿真。
2. `run_active_remaining.py`、`run_active_pour_video.py` 等是本地 WSL 编排器，
   含 PowerShell、Windows 可执行文件路径及本地 GPU 等待逻辑，不要在服务器直接运行。
   队列负责单卡分配和步骤顺序，不能用本地“所有GPU空闲”的等待逻辑。
3. `reconstruct_surface_snapshot.py` 已在提交历史中，当前为 Windows pysplashsurf 路径。
   将调用接到服务器可执行文件和 Linux 路径，保留25次平滑及其余重建参数。
4. `render_active_pour.py` 是 Blender 原生渲染脚本；Linux 适配需在 open_mainfile 之后
   调用资产包 support/active_asset_bundle.py 的 remap_loaded_images(asset_root)，
   避免 Y:/scenes HDRI 丢失；原文件不保存。使用现有 blender_single_gpu.py 隔离单卡。
5. 保留 `run_active_pour_video.py` 中的完整性检查：completed、301帧原生缓存、
   外观/几何哈希配对、30fps时间轴、原生同帧装置姿态、错误日志检查；不能只凭进程退出0。
   1280×960、64 samples、25次表面平滑，输出300帧/10秒MP4，ffprobe校验。
6. 仿真/重建/渲染/编码分明确 argv 步骤写入新 pipeline_job。
   结果留下简短总结、参数与产物清单；失败时保留缓存，不伪造完整成片。

本地未提交的 blender_coupled_event_overlay.py 改动属于独立的跨帧几何缓存优化，
本次未纳入；主动驱动渲染只导入其已有 _read_obj 和 _water_material，不依赖该优化。
不需复制或重建全部 scenes，也不需重新制作主动驱动外观。
