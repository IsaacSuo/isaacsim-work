# PhysX 诊断工具

这里存放需要 Isaac Sim / PhysX 运行时、但不属于生产仿真入口的碰撞诊断工具。

- `render_sdf_collision_preview.py`：单独生成并渲染刚体 SDF 碰撞表示。
- `run_penetration_audit.py`：启动 Blender 几何审核器并读取结果 JSON。退出码
  0 表示通过，1 表示工具或报告无效，2 表示审核有效但存在物理失败。

正式审核读取导出的 Visual、CollisionTetSurface 和
SimulationTetSurface，不依赖 PhysX 接触报告。通用数值判断复用
`soft_body/penetration.py` 与 `soft_body/tet_quality.py`，诊断工具不得被生产
入口导入，避免审核逻辑改变正式仿真状态。

示例：

```powershell
python tools/physx/run_penetration_audit.py `
  output/run/deformable_collision_debug_frame60.usdc `
  output/penetration_audits/run_frame60.json `
  --debug-report output/run/run_complete.json
```

默认阈值严格：穿透 1 mm、反转 Tet 0 个、非流形边 0 条。可用
`--support-y` 增加支撑平面审核。JSON 是主要结果，包含 `valid`、`passed`、
各层检查和失败列表。当前算法要求两表面发生精确交叉，不检查“闭合物体完全
包含另一闭合物体但表面没有交叉”的极端情形；报告会明确记录该限制。
