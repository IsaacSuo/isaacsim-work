# Conditioned inlet (opt-in)

## 服务器交接：当前采用 v90

这是本地已经跑完并渲染检查过的版本：**80 mm 上游导流段 + 越顶粒子回收 + 出口以上隐藏**。
不是早期的单层松弛或无导流壁实验。默认旧场景不受影响，只有显式配置该入口才启用。

完整可迁移预设在 `configs/conditioned_inlet_v90.json`，无需本地 v74/v90 输出缓存。
在仓库根目录执行下面的配置生成命令；将两个资源路径换成服务器实际位置：

```bash
python3 experiments/coupled_scenes/prepare_conditioned_inlet_preview.py \
  --scenes-root /path/to/scenes \
  --isaac-python /path/to/isaacsim/python.sh \
  --output output/coupled_scenes/v90_server_check
```

这一步只生成 `glass_cabinet_pour.json`、`bodies.json`、`run_command.json`，
**不会启动 GPU 仿真**，也不会覆盖已有输出目录。最后打印可执行的仿真命令；
服务器 agent 也可以直接按 `run_command.json` 的 `command` 参数数组启动。
场景、模型、已有碰撞 USD 和 Isaac 环境由服务器端准备；本次同步不包含这些资源。
尤其要确认命令中的 `--prebuilt-collision-usd` 指向服务器有效的 warehouse 碰撞缓存。

先复现短版，再加长，不要同时调整入口尺寸、速度和迭代数：

- 粒子间距 4 mm；粒子、软体 position iterations 均为 64。
- 输出/物理帧基准 60 Hz，每帧 12 个 substeps，即求解 720 Hz。
- 出生速度向下 0.96 m/s，`continuous_inlet`，8192 粒子一组的有界追加。
- PCO 为 0.003367003367003367 m；其他已验证碰撞设置保留。
- 入口内部净宽 88 × 64 mm；出生点在出口上方 80 mm，顶部再留 40 mm。
- 短版共 198 帧：120 帧物体预热、60 帧注水、18 帧尾段。
  名义出生 75,600 粒子，4.8384 L；不是所有出生水都已经穿过出口。
- 若需要约 30 L **出生量**，生成配置时追加 `--inlet-frames 372 --post-frames 30`。
  对应 468,720 粒子、29.99808 L、共 522 帧；出口实流量和柜内存水量应另行统计。

这个预设是 warehouse 玻璃柜内的**单个软体大象**（最大尺寸 0.55 m）诊断场景，
不是已经验证完毕的完整 mixed-material 多物体场景。柜子、物体位置保持本地验证值。
本地软件为 Isaac Sim 6.0.1 rc7 / Omni PhysX 110.1.13；其他版本仍需短版回归。

### 渲染：出口以上隐藏

沿用现有完整粒子缓存 → Splashsurf → Blender 流程。对 Blender 设置：

```text
COUPLED_EVENT_CONFIG=<本次输出目录>/glass_cabinet_pour.json
COUPLED_FLUID_SURFACE_DIR=<本次完整 Splashsurf 表面目录>
COUPLED_HIDE_UPSTREAM=1
RENDER_SCENE_NAME=warehouse
```

配置生成器已将除表面目录外的变量写入 `run_command.json.render_environment`，
需要由渲染启动方实际传入环境。隐藏的是导流外壳及出口平面以上的水材质（包括吸收体积），
不是删掉仿真粒子；不要在表面重建前过滤上游粒子，也不要切网格补盖。
这是全局出口高度裁切：下游水花如果再次飞到该平面以上，同样不显示，但仍参与仿真。
物理回收则不同：一旦曾经从出口排出，该粒子永久豁免上游回收。

### 已验证结果与边界

本地短版出生 75,600 粒子，回收 649 粒子（约 0.86%，0.041536 L），剩余 74,951。
流体、接触连续性、四面体轨迹门禁通过，未发现四面体翻转或柜壁/底板穿漏；
这不代表已经根治全部软体抖动，也不代表长时间、大水量运行已经验证。
服务器检查 `run_complete.json`、`primary_fluid/manifest.json` 和轨迹审计结果。
回收明细记录在 `primary_fluid/upstream_removals.jsonl`；缓存稳定粒子 ID 允许有缺号。

## Implementation contract

The default pour is unchanged. A `conditioned_inlet` object in the event JSON
adds four static guide walls and an upstream particle sink. Example:

```json
{
  "conditioned_inlet": {
    "outlet_centre": [0.011443081937356714, 0.13999692718812784, -8.27823311745492],
    "upstream_height": 0.08,
    "headroom": 0.04,
    "inner_size_xz": [0.088, 0.064],
    "wall_thickness": 0.01
  }
}
```

Set `source.centre` to outlet centre plus `[0, 0.08, 0]`; this is checked.
The validated diagnostic settings use `source.particle_contact_offset` =
`0.003367003367003367`, spacing 0.004, velocity `[0, -0.96, 0]`, 720 Hz,
64 particle iterations, and `chunked_density_sets` with capacity 8192.
Rigid guide contacts use rest offset 0, contact offset 0.001, restitution 0.

After each fetched substep, `PbdPourEvent.after_substep` removes particles above
the open guide top **only if they have never discharged below the outlet**.
No replacement water is emitted. Downstream splashes remain physical even if
they return above the guide. Removal compacts affected native USD particle sets;
it preserves surviving positions, velocities, density-derived per-particle mass,
and application-level stable IDs, verified immediately after authoring updates.
This can rewrite a previously full chunk; the bounded-append optimization is
not a promise that old chunks are never touched by the sink.

The fluid manifest includes cumulative removal/discharge statistics and
`upstream_removals.jsonl` records removed IDs and positions. Frame NPZ caches
contain only surviving particles and may therefore have gaps in particle IDs.
The Blender overlay renders the same four opaque walls. Fluid reconstruction
uses all cached particles, including the upstream segment; no render clipping
or synthetic nozzle-end surface is applied.

For a render-only hidden inlet, set `COUPLED_HIDE_UPSTREAM=1`. This omits the
guide shell and makes water surface/volume absorption transparent above the
physical outlet world-Y plane (Blender world Z). The full mesh and simulation
cache remain untouched; no cap surface or particle deletion is introduced.

This is a numerical inlet with an explicitly non-conservative upstream sink,
not a closed-system fluid simulation. Short diagnostic success does not alone
validate long-duration coupled motion.
