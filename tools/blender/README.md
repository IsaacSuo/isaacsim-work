# Blender 渲染工具

- `render_blender_soft_body_cache.py`：把 Isaac 动画 USD 导入原始 `.blend`，恢复材质、灯光、HDRI 和相机并用 Cycles 渲染。
- `render_blender_selected_views.py`：批量渲染选定静帧。
- `render_blender_selected_videos.py`：批量渲染并用 FFmpeg 编码最终视频。
- `render_blender_scene_batch.py`：场景候选视角批处理。
- `audit_simulation_penetration.py`：只读导入调试 USD，用真实三角面交叉、Tet
  反转和表面拓扑审核仿真结果；完整结果写入 JSON。
- `render_*_reference.py`：特定场景的只读参考渲染。

最终相机配置位于 `configs/blender_camera_selections.json`。

直接调用审核脚本时必须给 Blender 加 `--python-exit-code 1`。此时进程 0
表示通过，进程 1 表示样本不合格或工具错误，以 JSON 的 `valid` 字段区分。
批处理建议调用 `tools/physx/run_penetration_audit.py`，它提供稳定的 0/1/2
退出码。
