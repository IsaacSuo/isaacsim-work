# Isaac Sim / PhysX 仿真工作区

这是一个面向物理与视觉数据实验的 Isaac Sim 6.0 工作区，包含 PhysX 液体、布料、体积软体、多场景碰撞，以及 Blender Cycles 最终渲染。生成结果统一写入 `output/`，源码与实验配置由 Git 管理。

## 正式入口

### 流体与长视频

- `physx_realistic_liquid.py`：PhysX PBD 流体主场景。
- `liquid_video_pipeline.py`：长视频缓存、分段渲染和编码的控制入口。
- `run_realistic_liquid_validation.bat`：短实验与回归验证。
- `run_realistic_liquid_long_video.bat`：流体长视频入口。
- `LONG_VIDEO_PIPELINE.md`：运行与恢复说明。

### 布料与软体

- `cloth_flag_hero.py` / `render_cloth_flag_video.bat`：布料旗帜示例。
- `soft_body_bounce_hero.py` / `render_soft_body_bounce_video.bat`：体积软体主入口。
- `render_all_static_scene_videos.bat`：按场景配置批量生成 Isaac 侧软体缓存或视频。
- `tools/blender/render_blender_selected_videos.py`：使用选定相机在 Blender Cycles 中渲染最终视频。

## 多场景软体链路

```text
configs/scene_experiments.json
        │
        ▼
tools/scenes/run_static_scene_videos.py
        │  调用 PhysX
        ▼
soft_body_bounce_hero.py ──► animated USD cache
                                   │
configs/blender_camera_selections.json
                                   │
                                   ▼
tools/blender/render_blender_selected_videos.py
        │  调用 Blender Cycles + FFmpeg
        ▼
output/blender_scene_videos/videos/*.mp4
```

环境 USD 只负责静态场景和精准碰撞；PhysX 负责运动；原始 `.blend`、统一 HDRI、场景灯具和 Cycles 负责最终视觉。

## 目录结构

```text
isaacsim_work/
├── assets/                   # 小型受控模型资产
├── configs/                  # 正式场景与相机配置
├── docs/                     # 架构和管线文档
├── experiments/
│   ├── omniglass/            # OmniGlass / RTX 材质实验
│   └── scenes/               # 一次性场景构建实验
├── tools/
│   ├── audit/                # Blender、USD、材质、灯光和几何诊断
│   ├── blender/              # Blender Cycles 渲染与视频批处理
│   ├── scenes/               # 多场景准备、预览和 PhysX 批处理
│   ├── probes/               # 底层 API 能力探针
│   └── postprocess/          # 独立后处理
├── archive/source_snapshots/ # 历史源码快照
├── tests/                    # 无 GUI 自动化测试
└── output/                   # 生成结果，Git 忽略
```

各工具目录的职责和调用方式见对应 `README.md`。更完整的依赖与技术债见 [`docs/SOURCE_ARCHITECTURE.md`](docs/SOURCE_ARCHITECTURE.md)。

## 版本控制与可复现性

- `output/`、日志和 Python 缓存不进入 Git。
- 文本源码固定为 LF，保证 Windows、WSL、Linux 和 CI 中字节一致。
- 长视频任务会记录生产脚本 SHA-256；根目录正式入口保持兼容。
- 不要手工修改任务目录中的 `job.json`、manifest 或完成标记。
