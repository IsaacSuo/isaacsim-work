# 固定拓扑网格长视频管线

这套管线将软体物理和 PathTracing 渲染分离。当前第一个适配器是 `soft_body_bounce_hero.py`；以后布料可以复用相同缓存格式。

## 执行顺序

```text
init-soft-body
→ simulate（连续 PhysX，只写网格缓存）
→ validate-cache
→ render-pilot 或 render-all（不再运行 PhysX）
→ encode
```

固定拓扑只保存一次三角形索引，每一帧保存：

- 局部空间顶点 `points`
- 根节点位移 `translate`
- 包围盒和时间轴元数据
- 缓存文件 SHA-256

## 初始化软体任务

在 WSL 中运行：

```bash
python3 /mnt/y/isaacsim_work/fixed_topology_video_pipeline.py init-soft-body \
  --job-id soft_elephant_2p5s_v1 \
  --duration 2.5 \
  --fps 60 \
  --physics-fps 60 \
  --width 960 --height 960 \
  --spp 32 \
  --segment-frames 50
```

物理参数可直接作为初始化参数修改，例如：

```bash
  --youngs-modulus 90000 \
  --linear-damping 1.1 \
  --density 1050 \
  --drop-height 3.2 \
  --deformable-resolution 24
```

每组物理参数必须使用新的 `job-id`，不能覆盖已有任务。

## 仿真与校验

```bash
python3 /mnt/y/isaacsim_work/fixed_topology_video_pipeline.py simulate \
  --job /mnt/y/isaacsim_work/output/deformable_video/soft_elephant_2p5s_v1

python3 /mnt/y/isaacsim_work/fixed_topology_video_pipeline.py validate-cache \
  --job /mnt/y/isaacsim_work/output/deformable_video/soft_elephant_2p5s_v1 \
  --full
```

`simulate` 通过 Windows 桥接启动 Isaac Sim。适配器复用现有 hero 的场景和 PhysX 循环，但不把逐帧 PathTracing 作为视频来源；它将运动写入固定拓扑缓存。

## 试渲染与完整渲染

```bash
python3 /mnt/y/isaacsim_work/fixed_topology_video_pipeline.py render-pilot \
  --job /mnt/y/isaacsim_work/output/deformable_video/soft_elephant_2p5s_v1
```

完整渲染要求显式确认帧数。默认 2.5 秒、60 fps 是 150 帧：

```bash
python3 /mnt/y/isaacsim_work/fixed_topology_video_pipeline.py render-all \
  --job /mnt/y/isaacsim_work/output/deformable_video/soft_elephant_2p5s_v1 \
  --confirm-production 150-frames
```

如已完成渲染但尚未编码：

```bash
python3 /mnt/y/isaacsim_work/fixed_topology_video_pipeline.py encode \
  --job /mnt/y/isaacsim_work/output/deformable_video/soft_elephant_2p5s_v1
```

## 与流体长视频的关系

- 两套管线共享“不可变任务、来源哈希、分段渲染、accepted 标记、PNG 校验和 H.264 编码”的原则。
- 流体使用每帧可变拓扑表面缓存。
- 软体/布料使用一次拓扑加每帧顶点缓存。
- 现有流体任务和脚本没有被修改，可以继续按原哈希运行。
