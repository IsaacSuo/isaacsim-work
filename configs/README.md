# 配置

- `scene_experiments.json`：每个静态场景的 PhysX 落点、碰撞、材质和灯光参数。
- `blender_camera_selections.json`：最终 Blender 相机、目标点和场景选择。
- `model_material_experiments.json`：模型、物理软硬度与 Blender 外观材质的试验矩阵。

配置是正式实验输入，应随源码提交。运行结果和临时覆盖配置写入 `output/`。

全部相机场景（包括 `apartment`）都必须存在于场景实验配置中，并提供 `local_collision_bounds`。个别细长或凹形模型可在模型实验中设置 `collision_contact_offset` 与 `collision_rest_offset`，补偿 PhysX 体积碰撞壳和可见表面的包络差异。
