# 配置

- `scene_experiments.json`：每个静态场景的 PhysX 落点、碰撞、材质和灯光参数。
- `blender_camera_selections.json`：最终 Blender 相机、目标点和场景选择。

配置是正式实验输入，应随源码提交。运行结果和临时覆盖配置写入 `output/`。

`apartment` 使用单独验收过的缓存与灯光设置，因此只出现在相机配置中；其余相机场景都必须存在于场景实验配置中。
