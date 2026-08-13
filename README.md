# Isaac Sim / PhysX 仿真工作区

这个目录同时包含流体、长视频缓存渲染、布料和软体实验。当前采用“入口保持原位、文档明确分层”的整理方式，以免破坏批处理中的绝对路径和长视频任务记录的源码哈希。

## 从哪里开始

### 流体主线

- `physx_realistic_liquid.py`：当前主场景。负责 PhysX PBD 粒子、喷流/回收、原生 isosurface、场景构建、物理指标和缓存写入。
- `run_realistic_liquid_validation.bat`：短实验与回归验证入口。
- `liquid_video_pipeline.py`：长视频任务编排入口；支持初始化、仿真、缓存校验、分段渲染、编码和状态查询。
- `LONG_VIDEO_PIPELINE.md`：长视频运行说明。

### 长视频管线模块

- `liquid_video_cache.py`：任务配置校验、来源哈希、NPZ 表面缓存、manifest、PNG 校验和原子写入。
- `render_realistic_liquid_cache.py`：从缓存重建 USD 网格并分段渲染。
- `encode_realistic_liquid_video.py`：严格校验帧后调用 FFmpeg 编码。
- `run_realistic_liquid_stage.ps1`：Windows/Isaac Sim 阶段执行、日志和显存验收。
- `run_realistic_liquid_long_video.bat`：从 Windows 转入 WSL 编排器的通用入口。

### 布料与软体

- `cloth_flag_hero.py` / `render_cloth_flag_video.bat`：表面可变形旗帜仿真与视频。
- `soft_body_bounce_hero.py` / `render_soft_body_bounce_video.bat`：体积可变形模型跌落、压缩、反弹与视频。

## 源代码分层

| 层级 | 文件 | 建议 |
| --- | --- | --- |
| 正式管线 | `physx_realistic_liquid.py`、`liquid_video_*.py`、`render_realistic_liquid_cache.py`、`encode_realistic_liquid_video.py` | 保持路径稳定；生产任务会记录这些文件的 SHA-256 |
| 正式示例 | `cloth_flag_hero.py`、`soft_body_bounce_hero.py` | 可继续维护；后续可抽取共用的场景/渲染工具 |
| 基线/旧实现 | `physx_fluid_official.py`、`physx_water_clean.py`、`physx_official_baseline.py` | 保留用于行为与材质对照，不作为新功能入口 |
| 诊断探针 | `physx_*_probe.py`、`cloth_api_probe*.py` | 一次性 API/能力验证；不属于生产管线 |
| 历史快照 | `backups/`、`*.bak_*` | 仅用于回溯；不要从这里启动新任务 |
| 外部后处理 | `reconstruct_splashsurf.py` | 独立的 Splashsurf 网格重建工具，当前长视频主线不引用 |

更详细的依赖、风险和重构顺序见 [`docs/SOURCE_ARCHITECTURE.md`](docs/SOURCE_ARCHITECTURE.md)。

## 输出目录

`output/` 是可再生或半可再生的运行产物，不是源码。主要包括：

- `output/long_video/`：正式缓存、渲染帧和视频。
- `output/real_jet_validation_v1/`：喷流参数开发与验收。
- 其余大量目录：材质 A/B、相机、发射器和几何探针。

不要手工改写长视频任务内的 `job.json`、manifest、`*_complete.json` 或 `*_accepted.json`；它们共同构成可复现性与恢复机制。

## 当前约束

- 多个启动脚本硬编码 `Y:\isaacsim`、`Y:\isaacsim_work` 和 `/mnt/y/isaacsim_work`。
- `physx_realistic_liquid.py` 在模块级解析参数并执行 Isaac Sim，暂时不能作为普通库安全导入。
- 已初始化的长视频任务绑定了源码哈希；修改正式管线后，旧任务可能只能用对应快照继续。
- 此目录当前不在 Git 仓库中；`.gitignore` 已准备好，便于后续初始化版本控制。

