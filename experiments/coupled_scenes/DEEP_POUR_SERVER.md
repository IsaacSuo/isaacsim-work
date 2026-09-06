# Glass-cabinet deep-pour server handoff

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

The preset simulates 510 physics frames at 60 output Hz with four substeps:

- Frames 1-120: dry body prewarm.
- Frames 121-420: 0.9 m/s continuous inlet through a 42 by 30 particle-column
  nozzle (0.168 m by 0.120 m).
- Frames 421-510: post-inlet coupling and settling.
- Suggested video capture: physics frames 120-510, stride 2 for 30 fps.

The full job contains the original two rigid bodies and one deformable body.
For a first server-side isolation check, add `--deep-pour-deformable-only`; it
matches the locally validated one-deformable-body contact topology. The full
mixed 1.42-million-particle run still needs its first server validation.

Do not reduce these production contact settings during that validation:

- particle and deformable position iterations: 32 / 32;
- particle and deformable maximum depenetration velocity: 0.25 m/s;
- deformable volume contacts: 16,777,216;
- deformable surface contacts: 4,194,304;
- particle contacts: 8,388,608;
- collision stack: 2,147,483,648 bytes.

Treat a run as usable only if the generated manifest is complete and valid,
no GPU contact buffer reaches its capacity, no PhysX contact overflow is
reported, and the deformable trajectory/contact audits pass. Render after
those gates rather than using a partially completed cache.
