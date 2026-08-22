# 审计工具

本目录中的脚本用于只读检查 Blender、USD、HDRI、材质、灯光、相机、碰撞网格和渲染帧。它们可以单独运行，但正式仿真与渲染脚本不应导入这里的代码。

脚本名前缀表示用途：`audit_` 生成汇总，`inspect_` 检查单项，`measure_` 做场景测量，`probe_` 做受控实验，`validate_` 和 `verify_` 执行配置验收。

`audit_cache_rigidity.py` 依赖 USD/Kit，须用 Isaac Sim Python 运行；它先剔除整体平移与旋转，再计算固定拓扑缓存相对初始形状的归一化 RMS 形变。

- `audit_scene_collision_assets.py`：检查 14 个场景落点的真实网格高度、三角面朝向及 5×5 足迹覆盖。
- `summarize_scene_physics_validation.py`：把几何审计与 90 帧 PhysX 回归合并，检查冲击点相对真实表面的偏差。
