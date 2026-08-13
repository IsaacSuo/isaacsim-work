# Isaac Sim / PhysX 仿真工作区

这是一个包含 PhysX 液体、长视频缓存渲染、布料和软体实验的 Isaac Sim 6.0 工作区。正式运行入口保持在根目录；一次性探针、独立工具和历史快照已经分层收纳。

## 正式入口

### 流体与长视频

- `physx_realistic_liquid.py`：PhysX PBD 流体主场景，包含喷流、粒子回收、isosurface、物理指标和缓存写入。
- `run_realistic_liquid_validation.bat`：短实验、A/B 和回归验证。
- `liquid_video_pipeline.py`：长视频任务编排，支持初始化、仿真、缓存校验、分段渲染和编码。
- `liquid_video_cache.py`：长视频任务的数据契约、NPZ 缓存、manifest、来源哈希与完整性校验。
- `render_realistic_liquid_cache.py`：缓存到 USD 网格的分段渲染器。
- `encode_realistic_liquid_video.py`：严格帧校验和 FFmpeg 编码器。
- `run_realistic_liquid_stage.ps1`：Windows/Isaac Sim 阶段执行与日志、显存验收。
- `LONG_VIDEO_PIPELINE.md`：长视频操作说明。

### 可变形体示例

- `cloth_flag_hero.py` / `render_cloth_flag_video.bat`：表面可变形旗帜。
- `soft_body_bounce_hero.py` / `render_soft_body_bounce_video.bat`：体积可变形软体跌落与反弹。

## 目录结构

```text
isaacsim_work/
├── assets/                    # 小型、受版本控制的模型资产
├── archive/source_snapshots/  # 历史源码快照，只用于回溯
├── docs/                      # 架构和整理说明
├── tools/
│   ├── probes/cloth/          # 布料/可变形 API 探针
│   ├── probes/physx/          # 粒子与底层 PhysX API 探针
│   └── postprocess/           # 独立外部后处理工具
├── output/                    # 仿真、缓存和渲染产物；Git 忽略
└── *.py / *.bat / *.ps1       # 正式入口、基线和兼容启动器
```

## 代码分层

| 层级 | 位置 | 使用方式 |
| --- | --- | --- |
| 正式生产管线 | 根目录的 `physx_realistic_liquid.py`、`liquid_video_*`、渲染器和编码器 | 保持文件名和参数兼容，任务会记录 SHA-256 |
| 正式示例 | 根目录的 `cloth_flag_hero.py`、`soft_body_bounce_hero.py` | 可继续维护和运行 |
| 基线/旧实现 | `physx_fluid_official.py`、`physx_water_clean.py`、`physx_official_baseline.py` | 用于官方行为、材质和相机对照 |
| 诊断探针 | `tools/probes/` | 单独运行，不从生产脚本导入 |
| 独立后处理 | `tools/postprocess/` | 当前长视频主线不依赖 |
| 历史快照 | `archive/source_snapshots/` | 只读回溯，不作为新任务入口 |

详细依赖和后续重构顺序见 [`docs/SOURCE_ARCHITECTURE.md`](docs/SOURCE_ARCHITECTURE.md)。

## 版本控制与可复现性

- Git 跟踪源码、文档、启动器、历史快照和小型 assets。
- `output/`、日志和 Python 缓存不会进入 Git。
- 文本源码固定为 LF，以保证 Windows、WSL、Linux 和 CI 中字节一致。
- 不要手工修改长视频任务中的 `job.json`、manifest、`*_complete.json` 或 `*_accepted.json`。
- 未完成的长视频任务绑定了源码哈希；正式入口暂不移动或大规模格式化。

