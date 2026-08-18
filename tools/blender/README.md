# Blender 渲染工具

- `render_blender_soft_body_cache.py`：把 Isaac 动画 USD 导入原始 `.blend`，恢复材质、灯光、HDRI 和相机并用 Cycles 渲染。
- `render_blender_selected_views.py`：批量渲染选定静帧。
- `render_blender_selected_videos.py`：批量渲染并用 FFmpeg 编码最终视频。
- `render_blender_scene_batch.py`：场景候选视角批处理。
- `render_*_reference.py`：特定场景的只读参考渲染。

最终相机配置位于 `configs/blender_camera_selections.json`。
