# 多模型与材质实验

`run_experiments.py` 在多个已验收场景中运行受控模型试验：PhysX 生成运动缓存，Blender Cycles 使用对应原始场景和已验收相机渲染关键帧。当前矩阵覆盖 13 个场景、28 组实验；除 hospital 外，每个场景至少包含一组透明硅胶和一组非透明硬质材质，便于做场景内外观与物理对照。

```powershell
Y:\isaacsim\python.bat experiments\model_material\run_experiments.py --only banana_silicone_soft
```

默认 `--render-frames auto` 会从物理报告中选择初始、接触、最大压缩和回弹帧。配置中的 `camera_distance_scale` 仅沿已验收视线缩短拍摄距离，不会改动物理落点。已有且参数完全匹配的 PhysX/Blender 缓存会自动复用；`--force` 才会强制重算。

试验矩阵位于 `configs/model_material_experiments.json`。透明白浊硅胶允许软、硬两档可变形参数；粗糙白膜和拉丝金属只允许高刚度物理配置。每个物理档位分别配置线性阻尼、稳定阻尼和接触恢复系数，避免统一的高稳定阻尼让物体触地后瞬间停止。当前硬质基线使用 900 MPa 的高刚度可变形体，适合保持视觉轮廓；它仍不是数学上的零形变刚体。

场景落点、精准局部碰撞和基线相机来自 `configs/scene_experiments.json` 与已有验收报告。`hospital`、`mountain` 会提取并复用局部真实网格碰撞；其他平坦落点使用有限地面碰撞，避免无限平板阻止物体离开支撑区域。

需要量化软硬差异时，用 `tools/audit/audit_cache_rigidity.py` 对固定拓扑缓存做刚体配准。该指标排除整体平移和旋转，报告剩余的非刚性顶点位移。

完整视频复用已完成的 PhysX 缓存，不会重新仿真：

```powershell
Y:\isaacsim\python.bat experiments\model_material\render_videos.py
```

PNG 序列、H.264 视频和汇总报告分别写入 `output/model_material_experiments/video_frames/`、`videos/` 和 `video_summary.json`。

## 多物体互碰

`soft_body_bounce_hero.py` 支持重复传入 `--secondary-body MODEL HEIGHT YAW X Y Z`，也支持通过 `--body-config` 为每个物体指定独立物理参数。每个物体会建立独立的 PhysX volume-deformable hierarchy，因此物体之间参与真实碰撞；`selfCollision` 仍只控制单个物体内部的自碰撞。导出的 Blender USD 为每个物体保留独立的定拓扑动画网格，便于分别赋材质。

运行报告中的 `interbody_contacts`、`detected_interbody_contact_pairs` 和 `all_bodies_penetration_valid` 分别记录物体对接触、接触对数量和全体地面穿透检查。普通实验仍可只记录接触；`--require-interbody-contact` 会把“至少一对物体发生接触”升级为缓存验收条件。

14 场景混合材质批次由 `run_multi_object_videos.py` 驱动。每个场景随机选择三个不同模型，并强制包含透明白浊硅胶、粗糙白膜、拉丝金属三种外观；三件物体采用近同轴错层落体，主动制造物体间碰撞。随机种子和最终选择写在 `configs/multi_object_scene_experiments.json`，默认种子为 `20260820`。

原始归档模型面数差异很大。第一次运行前先生成最多约五万面的仿真版本，原 STL 不会被修改：

```powershell
& 'D:\Program Files (x86)\Blender\blender.exe' --background --python tools\models\prepare_simulation_stls.py -- assets\archieved_models assets\simulation_ready_models 50000
Y:\isaacsim\python.bat experiments\model_material\generate_multi_object_config.py --seed 20260820
```

正式批次默认 300 帧、60 FPS、deformable resolution 24。建议先完成并验收物理缓存，再单独启动 Blender 渲染：

```powershell
Y:\isaacsim\python.bat experiments\model_material\run_multi_object_videos.py --stage physics
Y:\isaacsim\python.bat experiments\model_material\run_multi_object_videos.py --stage render
```

输出位于 `output/multi_object_mixed_material_14/`。每个缓存都会核对模型、材质、物理档位、逐物体碰撞偏移、精确场景碰撞资产以及互碰结果，参数不匹配时不会错误复用旧缓存。

冻结 benchmark 基础设施前的 M0 production baseline、严格验收条件和最终审计命令见
[`docs/M0_PRODUCTION_BASELINE.md`](../../docs/M0_PRODUCTION_BASELINE.md)。M0 不会把“USD 缓存可读但 PhysX 报告无效”视为成功。
