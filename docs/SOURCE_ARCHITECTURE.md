# 源代码架构与整理报告

更新时间：2026-08-13

## 总览

工作区根目录现有 27 个 Python 文件、9 个 BAT、1 个 PowerShell 脚本。代码实际上由四组组成，而不是一个单体项目：

1. PhysX 液体场景与喷流实验。
2. 10 秒长视频的缓存、分段渲染和编码管线。
3. 布料旗帜示例。
4. 软体模型跌落示例。

输出目录约 78.8 GB，但源码不到 1 MB。源码整理应优先解决入口、职责和可导入性，而不是重排输出帧。

## 核心依赖关系

```text
liquid_video_pipeline.py
├── liquid_video_cache.py
├── encode_realistic_liquid_video.py
├── run_realistic_liquid_stage.ps1
│   ├── physx_realistic_liquid.py
│   │   └── liquid_video_cache.py
│   └── render_realistic_liquid_cache.py
│       └── liquid_video_cache.py
└── output/long_video/<job>/job.json
```

`liquid_video_pipeline.py` 是控制面；Isaac Sim 中实际运行的是 `physx_realistic_liquid.py` 和 `render_realistic_liquid_cache.py`。`liquid_video_cache.py` 是两侧共享的数据契约。

## 文件职责

### 当前生产流体代码

#### `physx_realistic_liquid.py`

- 3286 行、69 个命令行参数，是当前最大且最关键的文件。
- 同时负责参数校验、喷射器规划、粒子系统、碰撞体、材质、灯光、相机、仿真循环、回收策略、喷流统计、渲染和缓存写入。
- 支持 `block`、`stream`、`emitter` 三种源模式。
- 长视频模式会核对自身 SHA-256 与 `job.json`，防止用错误版本续跑。
- 当前在模块级解析参数并启动 Isaac Sim；导入即产生副作用，不利于单元测试和复用。

#### `run_realistic_liquid_validation.bat`

- 为主流体脚本提供 `plan`、`smoke`、`ab`、`regression`、`path-preview`、`path-final` 和 `video-baseline` 阶段。
- 负责拒绝覆盖旧输出、采集 `nvidia-smi`、扫描致命日志和检查显存预算。
- 是短实验/回归入口，不是 10 秒生产任务的入口。

### 长视频代码

#### `liquid_video_cache.py`

- 纯 Python 数据层，是当前最适合测试和复用的模块。
- 管理不可变任务配置、simulation/render provenance、SHA-256、NPZ 表面网格、JSONL manifest、PNG 完整性与原子写入。
- `SurfaceCacheTakeWriter` 强制帧序号、物理步和时间轴连续。

#### `liquid_video_pipeline.py`

- WSL 侧编排 CLI。
- 命令：`init`、`status`、`simulate`、`validate-cache`、`render-segment`、`render-pilot`、`render-all`、`finalize-render`、`encode`。
- 将正式渲染固定拆成非重叠分段，并要求显式确认 300 帧生产渲染。

#### `run_realistic_liquid_stage.ps1`

- Windows 侧阶段包装器。
- 启动 Isaac Sim Python、记录日志和 GPU 采样，验证完成标记，然后写入 accepted 标记。

#### `render_realistic_liquid_cache.py`

- 读取 job 与缓存，在渲染模板中注入 `/World/CachedLiquid`。
- 支持按区间渲染、已验收帧恢复、临时文件原子替换和分段 manifest。
- 同样在模块级解析参数和启动 Isaac Sim。

#### `encode_realistic_liquid_video.py`

- 在编码前核对 PNG、尺寸、帧号、来源哈希和渲染 manifest。
- 输出 H.264/yuv420p/BT.709，并用 ffprobe/解码过程复核帧率、时长和帧数。

### 布料与软体

#### `cloth_flag_hero.py`

- 610 行；构建表面可变形旗帜、运动杆、灯光、相机、关键帧与可选逐帧视频。
- 有独立验收报告。

#### `soft_body_bounce_hero.py`

- 863 行；加载 STL，创建体积可变形层级，检测碰撞、压缩、横向膨胀和反弹。
- `soft_body_elephant_hero` 的旧报告因底面穿透阈值失败，后续输出是独立批次。

两个 hero 文件重复实现了 `stage_notice`、材质创建/绑定、立方体、look-at、矩形灯和截图等工具。后续应抽取 `sim_common/studio.py`，但这属于会改变源码哈希的重构，不应在仍需续跑旧长视频任务时仓促进行。

### 基线、探针和历史文件

- `physx_fluid_official.py`：最小官方风格流体基线。
- `physx_water_clean.py`：早期清水/连续发射实验，仍由 `run_official_fluid_headless.bat` 调用。
- `physx_official_baseline.py`：材质、isosurface、相机和渲染 A/B 基线，仍由 `run_official_baseline.bat` 调用。
- `physx_particle_recycle_probe.py`：粒子回收能力验证；不是生产入口。
- `physx_official_emitter_probe.py`、`physx_interface_capability_probe.py`：API 能力探针。
- `cloth_api_probe.py` 到 `cloth_api_probe8.py`：按探索顺序保留的 API 探针。
- `reconstruct_splashsurf.py`：外部 Splashsurf CLI 包装；当前主线不引用。
- `backups/` 和 `physx_realistic_liquid.py.bak_*`：历史源码快照。

## 主要代码问题

### 1. 主脚本职责过多

`physx_realistic_liquid.py` 将场景、物理、视觉、指标和 I/O 放在同一个模块中。建议最终拆成：

```text
liquid/
├── cli.py
├── scene.py
├── emitter.py
├── particles.py
├── metrics.py
├── rendering.py
└── cache_adapter.py
```

拆分时应先为 `validate_args`、喷射器布局、截面指标聚合和缓存契约补测试，再改变生产入口。

### 2. 导入副作用

多个脚本在模块级调用 `parse_args()`，随后直接初始化 `SimulationApp`。这使函数无法被普通测试进程导入。目标结构应是：

```python
def main(argv=None) -> int:
    args = parse_args(argv)
    # 此处之后才初始化 SimulationApp

if __name__ == "__main__":
    raise SystemExit(main())
```

### 3. 固定盘符

代码和启动器中存在 `Y:\isaacsim_work`、`Y:\isaacsim`、`/mnt/y/isaacsim_work`。短期保留以兼容现有机器；长期应由一个 `.env`/配置文件或 `ISAACSIM_ROOT`、`ISAACSIM_WORK_ROOT` 环境变量统一提供。

### 4. 历史快照没有版本控制

当前目录不是 Git 仓库，因此只能依靠 `backups/`、文件名时间戳和 job 中的 SHA-256 追踪版本。建议在排除 `output/` 后初始化 Git；生产任务仍继续保存源码哈希。

### 5. 文档编码

原 `LONG_VIDEO_PIPELINE.md` 中分辨率乘号和弯引号发生乱码，本次会统一修正为 UTF-8 可读字符。

## 推荐重构顺序

1. 先初始化 Git，只跟踪源码、文档、启动器和小型 assets；排除 `output/`。
2. 将模块级 CLI 改为 `main(argv=None)`，但保持原文件名和参数兼容。
3. 为 `liquid_video_cache.py` 和纯数学喷流函数建立测试。
4. 抽取布料/软体共用 studio 工具。
5. 拆分 `physx_realistic_liquid.py`，保留同名薄入口，以兼容批处理。
6. 最后再把 `cloth_api_probe*` 和 PhysX probes 移入 `tools/probes/`，并把历史快照统一到 `archive/source_snapshots/`。

## 为什么本次不直接移动正式文件

长视频 `job.json` 保存固定脚本名和 SHA-256，批处理与文档又使用绝对路径。直接移动或大规模格式化会让未完成的 `faucet_10s_v1` 与 `faucet_10s_v1_laminar` 无法按原 provenance 继续。当前整理先建立清晰入口和边界，后续重构应在冻结旧任务或保存对应源码快照后进行。

