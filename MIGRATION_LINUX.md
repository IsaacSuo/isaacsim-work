# Linux 服务器迁移与接管说明

本目录是从 Windows 工作区迁出的可运行快照，目标是让服务器端 agent 在不依赖原有盘符的情况下继续 Isaac Sim 6.0 物理实验和 Blender Cycles 渲染。归档保留 Git 历史及打包时所有未提交源码，也保留后续实验必需的模型、14 个场景、HDRI、精准碰撞资产和少量场景验收 JSON；大体积视频、PNG、日志和物理动画缓存未迁移。

## 1. 归档内容

- `.git/`：历史与迁移时的 dirty worktree，服务器接管后先运行 `git status --short`，不要用 reset 覆盖未提交工作。
- 根目录、`configs/`、`experiments/`、`tools/`、`tests/`、`docs/`、`soft_body/`、`archive/`：源码、配置、测试和少量历史参考。
- `assets/archieved_models/`：原始模型库；`assets/simulation_ready_models/`：约五万面以内的 PhysX 仿真版本。
- `scenes/`：14 个正式 `.blend`、各场景的 `.usdc` 与 `_sim.usda`、贴图目录、5 个统一 HDRI 以及 `static_scene_manifest.json`。
- `output/scene_collision_assets/`：从真实场景网格提取的局部精准碰撞资产。
- `output/blender_scene_videos/.../run_complete.json`、`blender_render_report.json` 和 `output/scene_ground_sites.json`：已验收落点、相机、地面和碰撞元数据；不包含对应的大缓存和图像。
- `BUNDLE_MANIFEST.sha256`：包内逐文件校验值。归档外另有整个 `.tar.gz` 的 SHA-256 文件。

## 2. 解包和环境

建议解包到本地 Linux 文件系统（例如 NVMe 的 `/data/isaacsim_work`），不要直接在 SMB 挂载目录中运行 PhysX 或输出逐帧 PNG。

```bash
tar -xzf isaacsim_linux_migration_20260821.tar.gz -C /data
cd /data/isaacsim_work
cp .env.linux.example .env.linux
# 修改 ISAAC_PYTHON/BLENDER_BIN/FFMPEG_BIN 后：
source .env.linux
chmod +x "$ISAAC_PYTHON"
python3 tools/migration/verify_linux_bundle.py --runtime
```

四个重要变量：

- `ISAAC_PYTHON`：同版本 Isaac Sim 自带的 `python.sh`。物理脚本不要用系统 Python。
- `BLENDER_BIN`：Blender 可执行文件；若在 `PATH` 中可设为 `blender`。
- `FFMPEG_BIN`：FFmpeg 可执行文件；若在 `PATH` 中可设为 `ffmpeg`。
- `SCENES_ROOT`：本包默认是 `$PWD/scenes`，通常无需改变。

完整哈希检查会读取数 GB 数据，可在传输完成后运行一次：

```bash
python3 tools/migration/verify_linux_bundle.py --hashes
```

## 3. 首次接管顺序

```bash
git status --short
git log -1 --oneline
"$ISAAC_PYTHON" -m py_compile \
  soft_body_bounce_hero.py \
  experiments/model_material/run_experiments.py \
  experiments/model_material/run_multi_object_videos.py \
  experiments/model_material/render_videos.py
"$ISAAC_PYTHON" -m unittest -v tests.test_repository_layout
```

先用一个实验做短验证，不要直接启动 14 场景长批次：

```bash
"$ISAAC_PYTHON" experiments/model_material/run_experiments.py \
  --only banana_silicone_soft --stage physics --frames 30
```

验证通过后，当前 14 场景、三物体、混合材质主链路分两段运行：

```bash
"$ISAAC_PYTHON" experiments/model_material/run_multi_object_videos.py --stage physics
"$ISAAC_PYTHON" experiments/model_material/run_multi_object_videos.py --stage render
```

分段的好处是先验收 PhysX 报告和穿透/互碰结果，再占用 GPU 做 Cycles。只重渲染已有缓存可运行第二条；由于本迁移包刻意不携带旧动画缓存，服务器第一次通常必须先完成 physics。

单物体/材质矩阵入口：

```bash
"$ISAAC_PYTHON" experiments/model_material/run_experiments.py --stage physics
"$ISAAC_PYTHON" experiments/model_material/run_experiments.py --stage render
"$ISAAC_PYTHON" experiments/model_material/render_videos.py
```

## 4. Windows → Linux 风险清单

- 正式模型/材质实验入口已改为环境变量和项目相对路径。仓库仍保留一些历史 `.bat`、`.ps1` 及 legacy 工具中的 Windows 绝对路径；Linux 不应把这些当正式入口。
- 文件系统区分大小写。自检会报告只差大小写的路径冲突；后续新增资产时，JSON 中的文件名必须与磁盘完全一致。
- Git 文本规则固定为 LF。若从共享盘再次复制脚本，不要让客户端改回 CRLF。解包后的 shell 脚本若无执行位，可显式 `chmod +x`。
- `.blend` 中个别旧 World 节点仍记录 `Y:\scenes\HDRI\...`，且少数旧 HDRI 已不存在。这不影响正式链路：`render_blender_soft_body_cache.py` 会重建 World 并从 `$SCENES_ROOT/HDRI` 加载统一 HDRI。若人工直接打开 `.blend` 渲染，则需手工重绑 World 贴图。
- 场景材质贴图大多 packed 在 `.blend`；USD 侧仍保留各场景目录及贴图，不能只复制根目录 `.blend`。
- 旧 JSON 里保留 Windows 路径是正常的，跨平台入口会按最终资产文件名识别这些验收报告。新的服务器输出会记录 Linux 绝对路径。
- Isaac Sim 版本相同仍不代表 GPU 栈完全相同。首次运行应确认 NVIDIA 驱动、RTX/CUDA、Vulkan/EGL headless 和 OptiX；Blender 还需确认 Cycles device 不是意外回退到 CPU。
- 输出可能快速增长到几十或上百 GB。将 `output/` 放在高速本地盘，定期只保留报告、最终视频和确有复用价值的 USD 缓存。
- 物理使用 120 Hz/子步进、渲染降采样到较低 FPS 是可行策略，但现有批处理默认按缓存每帧渲染并以 60 FPS 编码；若更改采样率，务必同时记录 physics timestep、缓存帧率和编码 FPS，避免视觉时间尺度错误。

## 5. 目录职责和不要改动的基线

`configs/scene_experiments.json` 决定 14 个场景的落点、局部碰撞范围、相机及自由坠落策略；`configs/model_material_experiments.json` 定义软硅胶、硬质白膜和金属的物理/视觉配对；`configs/multi_object_scene_experiments.json` 固定随机种子和每个场景的三物体组合。未经新一轮验收，不要批量覆盖这些基线。

`hospital` 与 `mountain` 使用真实局部网格碰撞，不能退化为无限平板；其他场景也已针对参考地面穿模做过检查。金属和粗糙白膜仍由 PhysX 高刚度可变形体模拟，不是严格刚体。渲染侧材质由 Blender 脚本按每个缓存物体分别赋予。

服务器 agent 如需整理，优先保持正式入口、配置和资产相对路径稳定；可以清理新生成输出，但应先保存 `run_complete.json`、`blender_render_report.json`、批次 summary 以及用于复现的命令和环境变量。
