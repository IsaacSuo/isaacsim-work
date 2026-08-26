# 源代码架构与整理状态

更新时间：2026-08-26

## 当前代码线

仓库包含五条相对独立的代码线：

1. PhysX PBD 液体与喷流实验。
2. 可恢复的长视频缓存、分段渲染和编码管线。
3. 表面可变形布料。
4. 体积软体、多场景静态碰撞与动画 USD 导出。
5. Blender Cycles 材质、灯光、相机和最终视频渲染。

生成数据集中在 `output/`，不参与源码结构设计，也不进入 Git。

## 正式依赖

### 流体长视频

```text
liquid_video_pipeline.py
├── liquid_video_cache.py
├── encode_realistic_liquid_video.py
├── run_realistic_liquid_stage.ps1
│   ├── physx_realistic_liquid.py
│   └── render_realistic_liquid_cache.py
└── output/long_video/<job>/job.json
```

### 多场景软体与 Blender 视频

```text
configs/scene_experiments.json
└── tools/scenes/run_static_scene_videos.py
    └── soft_body_bounce_hero.py
        └── output/.../soft_body_blender.usdc

configs/blender_camera_selections.json
└── tools/blender/render_blender_selected_videos.py
    ├── tools/blender/render_blender_soft_body_cache.py
    ├── Blender scene.blend + Cycles
    └── FFmpeg
```

`soft_body_bounce_hero.py` 保持根目录兼容入口，因为固定拓扑管线和已有任务会记录它的文件名与 SHA-256。

## 分层规则

| 位置 | 职责 | 稳定性 |
| --- | --- | --- |
| 根目录正式脚本 | 用户入口、缓存协议、兼容启动器 | 保持名称与参数兼容 |
| `configs/` | 可审查、可复现的场景和相机参数 | 正式输入 |
| `tools/scenes/` | 多场景准备、PhysX 批处理和静态预览 | 正式工具 |
| `tools/blender/` | Blender 导入、材质、灯光、相机与编码 | 正式工具 |
| `tools/audit/` | 只读诊断、测量、对比和可视化检查 | 非生产依赖 |
| `tools/probes/` | Isaac/PhysX API 能力实验 | 非生产依赖 |
| `experiments/` | 保留的阶段性材质或场景实验 | 不作为正式入口 |
| `archive/` | 历史快照 | 只读回溯 |

## 主要技术债

### `physx_realistic_liquid.py`

- 3286 行，同时承担参数、场景、粒子、材质、指标、渲染和缓存职责。
- 模块导入时会解析参数并启动 `SimulationApp`。
- 受已有长视频任务源码哈希约束，拆分时必须保留薄兼容入口。

### `soft_body_bounce_hero.py`

- 当前约 3000 行，仍保留为兼容入口和 Isaac Sim 执行编排器。
- 不依赖 Isaac Sim 的逻辑逐步放入 `soft_body/`，目前已包含配置归一化、几何裁剪和四面体质量计算。
- 混合了参数解析、软体生成、环境材质修复、灯光恢复、局部精准碰撞、物理验收、截图和 Blender USD 导出。
- 拆分以纯函数和可单测边界为主，不改变根入口、命令行参数或已验收的物理默认值。

推荐后续拆成：

```text
soft_body/
├── config.py          # 物体配置归一化与基础约束
├── geometry.py        # 局部裁剪等纯几何逻辑
├── tet_quality.py     # Tet 体积、长宽比与反转统计
├── collision.py       # 后续：网格清理与精确碰撞
├── environment.py     # USD 环境、材质与灯光恢复
├── deformable.py      # 软体层级和 PhysX 参数
├── validation.py      # 后续：穿透、压缩、反弹与自由坠落验收
└── blender_export.py  # 固定拓扑动画 USD
```

根目录 `soft_body_bounce_hero.py` 最终只保留 CLI 和执行编排。

## 下一轮重构顺序

1. 基于 `soft_body/tet_quality.py` 建立独立的穿透与 Tet 反转审核模块。
2. 运行 garage、hospital、mountain 三类代表场景的缓存回归。
3. 再抽取 Blender USD 导出；环境材质与灯光保持一组，避免过度拆分。
4. 最后处理 `physx_realistic_liquid.py`，并保留旧任务可恢复性。
