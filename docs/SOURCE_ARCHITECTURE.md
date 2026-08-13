# 源代码架构与整理状态

更新时间：2026-08-13

## 当前代码线

工作区由四条相对独立的代码线组成：

1. PhysX 液体主场景和喷流实验。
2. 10 秒长视频缓存、分段渲染与编码管线。
3. 表面可变形布料旗帜。
4. 体积可变形软体跌落与反弹。

输出约 78.8 GB，源码不足 1 MB。源码整理优先关注入口、职责、依赖和可测试性；输出仅作为运行记录。

## 长视频依赖

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

`liquid_video_pipeline.py` 是控制面；Isaac Sim 中实际执行的是 `physx_realistic_liquid.py` 和 `render_realistic_liquid_cache.py`；`liquid_video_cache.py` 是两侧共享的数据契约。

## 正式源文件

### `physx_realistic_liquid.py`

- 3286 行、69 个命令行参数，是核心和最大文件。
- 同时处理参数、喷射器规划、粒子系统、碰撞体、材质、灯光、相机、回收、喷流统计、仿真循环、渲染和缓存。
- 支持 `block`、`stream`、`emitter` 三种源模式。
- 长视频模式会核对自身 SHA-256 与任务配置。
- 当前在模块级解析参数并启动 Isaac Sim，不适合作为普通库导入。

### `liquid_video_cache.py`

- 纯 Python 数据层，适合优先补测试。
- 管理不可变配置、provenance、SHA-256、NPZ 表面网格、JSONL manifest、PNG 校验和原子写入。
- `SurfaceCacheTakeWriter` 强制帧号、物理步和时间连续。

### `liquid_video_pipeline.py`

- WSL 侧编排 CLI。
- 命令包括 `init`、`status`、`simulate`、`validate-cache`、`render-segment`、`render-pilot`、`render-all`、`finalize-render` 和 `encode`。
- 正式渲染拆成非重叠分段，并要求显式确认生产帧数。

### `run_realistic_liquid_stage.ps1`

- Windows 侧包装器，启动 Isaac Sim Python。
- 采集日志和 GPU 使用，验证完成标记后写入 accepted 标记。

### `render_realistic_liquid_cache.py`

- 将缓存表面注入渲染模板的 `/World/CachedLiquid`。
- 支持分段恢复、临时文件原子替换和 manifest 验收。

### `encode_realistic_liquid_video.py`

- 编码前核对 PNG、尺寸、帧号、来源哈希和渲染 manifest。
- 编码后复核编解码器、帧率、时长、帧数和色彩标签。

## 可变形体代码

- `cloth_flag_hero.py`：610 行，构建表面可变形旗帜和运动杆。
- `soft_body_bounce_hero.py`：863 行，构建体积可变形层级并测量碰撞、压缩和反弹。

两个文件重复实现了材质创建/绑定、立方体、look-at、矩形灯、阶段日志和截图工具。后续可抽取 `sim_common/studio.py`。

## 已完成的目录整理

- 布料 API 探针：`tools/probes/cloth/`。
- 粒子与接口探针：`tools/probes/physx/`。
- Splashsurf 后处理：`tools/postprocess/`。
- 历史源文件：`archive/source_snapshots/`。
- 正式入口继续留在根目录，避免破坏批处理绝对路径和任务 provenance。

## 主要技术债

1. `physx_realistic_liquid.py` 职责过多。
2. 多个脚本在模块级解析 CLI 并初始化 `SimulationApp`，存在导入副作用。
3. 启动器硬编码 `Y:\isaacsim`、`Y:\isaacsim_work` 和 `/mnt/y/isaacsim_work`。
4. 布料与软体示例存在重复 studio 工具。
5. 纯数学和缓存契约尚无独立自动化测试套件。

## 推荐的下一轮顺序

1. 在不改变命令行参数的前提下，将 CLI 收口到 `main(argv=None)`。
2. 为 `liquid_video_cache.py` 和喷流纯数学函数建立测试。
3. 抽取布料/软体共用 studio 工具。
4. 将 `physx_realistic_liquid.py` 拆为 `liquid/scene.py`、`emitter.py`、`metrics.py`、`rendering.py` 和 `cache_adapter.py`。
5. 保留原 `physx_realistic_liquid.py` 作为薄兼容入口。

未完成的 `faucet_10s_v1` 与 `faucet_10s_v1_laminar` 仍绑定旧源码哈希。进行正式模块拆分前，应冻结这些任务或确保对应源码快照可供续跑。

