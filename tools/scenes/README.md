# 场景工具

- `prepare_static_scenes.py`：审计导出的 USD 并构建静态碰撞 wrapper。
- `run_static_scene_previews.py`：批量运行低成本预览或导出 Blender 动画缓存。
- `run_static_scene_videos.py`：批量运行 5 秒 PhysX 场景任务。
- `run_environment_panoramas.py`：不运行物理，只生成四方向环境预览。
- `extract_collision_asset.py`：按世界坐标边界裁剪原场景三角网格，统一承托面朝向并去除重复面，生成可复用的精准局部碰撞 USD。

模型/材质实验可使用 `run_experiments.py --stage collision` 只重建碰撞资产，不启动物理或渲染。生成物统一放在 `output/scene_collision_assets/`。

正式配置位于 `configs/scene_experiments.json`。根目录 `render_all_static_scene_videos.bat` 是兼容启动器。
