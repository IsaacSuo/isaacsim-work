# PhysX 诊断工具

这里存放需要 Isaac Sim / PhysX 运行时、但不属于生产仿真入口的碰撞诊断工具。

- `render_sdf_collision_preview.py`：单独生成并渲染刚体 SDF 碰撞表示。

后续的仿真穿透审核入口可以放在这里负责读取 USD/TetMesh；通用的数值判断应
复用 `soft_body/tet_quality.py`，结果统一写成 JSON。诊断工具不得被生产入口
导入，避免审核逻辑改变正式仿真状态。
