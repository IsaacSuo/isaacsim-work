# Glass-cabinet deep-pour server handoff

## 2026-09-10 默认水材质更新

默认涡量补偿现为 `source.vorticity_confinement = 10.0`。普通倾倒 builder、
运行入口缺省值、conditioned-inlet 服务器预设均已更新；其他物理参数不变。
最新入口布局与服务器执行说明以 [CONDITIONED_INLET.md](../../coupled_scene/CONDITIONED_INLET.md)
为准。已生成的任务 JSON 若显式保留 0.02，更新代码后仍会使用 0.02：应在
新任务配置中明确改成 10，保留服务器原来的水量、入口和显存配置。
本地短版已检查并渲染通过；不把它当成大水量/长时间验证。

## 最新交接：分块发射为可选方案，默认配置不变

先阅读 [分块实现与实测记录](EMITTER_CHUNK_BENCHMARK.md)。本次同步的是代码、
测试和说明，不包含本地仿真缓存、视频、场景或运行环境。

- 新策略 `chunked_density_sets`：多次发射共用一个有容量上限的粒子集，
  当前块装不下下一批时换块。旧块继续正常参与 PhysX 耦合，但不因加水而被重写。
- 对已有连续入口配置，设置 `particle_set_strategy` 为 `chunked_density_sets`，
  `emission_chunk_particles` 为 `8192` 即可复用本地测试的容量；省略容量时为 `16384`。
  这些是每块容量，不是总粒子上限，也不是一次喷出的粒子数。
- 本地 37,800 粒子、360 次发射被组织为 5 个块，重复处理旧数据量减少约 80%。
  缓存、粒子 ID、软体四面体和接触资源检查通过。尚未做新版视觉对比。
- **不能据此承诺整体加速百分比**：直接数组操作仅节省约 0.11 秒，
  两次预热耗时也有明显波动。百万粒子的服务器吞吐和稳定性仍需验证。
- builder 可加 `--emission-chunk-particles 8192` 显式启用。注意与
  `--server-deep-pour` 合用时，还会将旧的帧分层入口切换为 `continuous_inlet`。
  若做仅存储策略的 A/B，请以已有 `continuous_inlet` 配置为基线，其他参数不变。
- 同步还修正了旧显式质量路径：粒子 helper 接受的是**单粒子质量**，内部已乘粒子数，
  不能再传整批质量。旧多集实验中的质量被重复乘算，因此不能把那些崩溃直接解释成
  “粒子集数量达到引擎上限”。新分块版与共享密度版均沿用密度推导质量策略。

下文保留原服务器预设的用法；其默认仍是旧帧入口，不会自动启用本次分块实验。

## Original server preset

This preset keeps the original cabinet and object scale, uses 4 mm particles,
and raises the five-second inlet volume to a nominal 90.72 L. That is enough
for a 30.4 mm flat-water equivalent in the 2.0 m by 1.4909 m cabinet. The
depth is an injected-volume reference, not a promise that every particle will
remain in the cabinet after splashing.

Generate the full rigid/deformable job with:

```bash
python experiments/coupled_scenes/build_glass_cabinet_run.py \
  --scene warehouse \
  --server-deep-pour
```

The builder writes `run_command.json`, `bodies.json`, and
`glass_cabinet_pour.json` beneath
`output/coupled_scenes/warehouse_glass_cabinet_deep_pour_server_3cm_4mm`.
The command stored in `run_command.json` uses this workstation's Windows/WSL
paths; the server agent should translate only launcher, asset, scene, and
output paths for its local environment, while retaining the generated physics
and source settings.

The preset simulates 510 physics frames at 60 output Hz with twelve substeps
(720 Hz contact updates):

- Frames 1-120: dry body prewarm.
- Frames 121-420: 0.96 m/s continuous inlet through a 41 by 29 particle-column
  nozzle (0.164 m by 0.116 m), contributing exactly four 4 mm-spaced layers
  per 60 Hz output frame (91.3152 L / 1,426,800 particles in total).
- Frames 421-510: post-inlet coupling and settling.
- Suggested video capture: physics frames 120-510, stride 2 for 30 fps.

The full job contains the original two rigid bodies and one deformable body.
Before running it, launch the short production-contact test:

```bash
python experiments/coupled_scenes/build_glass_cabinet_run.py \
  --scene warehouse \
  --server-deep-pour \
  --deep-pour-contact-probe
```

This emits one pre-authored frame batch from the production-width nozzle with
the production 4.2 m/s native PhysX speed ceiling. It is intended to validate
the actual production contact path rather than make a final render. For a longer
one-deformable-body run, use `--deep-pour-deformable-only` instead. The full mixed
1.42-million-particle run still needs its first server validation.

Do not reduce these production contact settings during that validation:

- particle and deformable position iterations: 64 / 32;
- physics contact frequency: 720 Hz (12 substeps per 60 Hz frame);
- TGS maximum bias coefficient: 240, preventing the 720 Hz timestep from
  turning small constraint errors into arbitrarily fast penetration correction;
- deformable collision target: 20,000 surface triangles;
- pre-authored frame batches: the running simulation never resizes and
  re-authors the already-active particle set;
- particle and deformable maximum depenetration velocity: 0.25 m/s;
- production particle maximum speed: 4.2 m/s, derived from the inlet-to-floor
  ballistic speed with margin; more than 0.5% of active particles touching this
  ceiling invalidates the cache;
- deformable volume contacts: 16,777,216;
- deformable surface contacts: 4,194,304;
- particle contacts: 8,388,608;
- collision stack: 2,147,483,648 bytes.

Treat a run as usable only if the generated manifest is complete and valid,
no GPU contact buffer reaches its capacity, no PhysX contact overflow is
reported, and the deformable trajectory/contact audits pass. Render after
those gates rather than using a partially completed cache.
