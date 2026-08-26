# 工具目录

- `scenes/`：正式的多场景 Isaac/PhysX 编排。
- `blender/`：正式的 Blender Cycles 渲染和视频编码。
- `physx/`：PhysX 碰撞表示的独立预览与诊断工具。
- `audit/`：只读诊断与验收。
- `probes/`：底层 API 探针。
- `postprocess/`：独立后处理。

正式运行脚本可以依赖同一正式目录或仓库根模块，但不应依赖 `audit/` 与 `probes/`。
