# 第一组液面场景

五个布局、装置动作、碰撞几何和相机已搭建；原生动作入口已接通，
静水及其余四种效果的样片已完成（v4/v5）。下面保留各阶段操作和历史记录。
五个场景均放在现有 `Y:\scenes\warehouse.blend` 内，不修改原环境。

2026-09-10：用户认可 v11 液面余波与 v91 倾倒短测后，普通运行默认涡量为
**10.0**；4 mm / 720 Hz / 64 次迭代与 25 次表面平滑不变。
历史 `--decay-test` 单变量诊断仍固定 0.02 基线，`--decay-vorticity` 显式覆盖。
旧记录不修改。恢复旧静水到新默认时记录材质变化并重新验证就绪；不宣称
此前 v4/v5 已使用涡量 10，也不预设新参数必能通过静置门槛。

服务器可使用原生 probe 与场景构建代码，但这里的串行启动/桌面交付脚本
是本机 Windows/WSL 编排，含本地路径，不能原样当成 Linux 服务器启动器。
服务器自行准备环境和布局资源、转换路径后调用原生入口；本次不上传资产缓存。

| 场景 | 装置动作 | 观察窗口 |
| --- | --- | --- |
| 静水 | 无 | 6 秒 |
| 容器晃动 | 2–6 秒沿 X 左右平移，幅度包络上限 6 cm | 10 秒 |
| 自由波浪传播 | 中央压头在 2–2.6 秒下压、退出 | 6 秒；只选边界回波返回前的片段研究自由传播 |
| 液面传播与反弹 | 端部推板在 2–2.8 秒前推 5.5 cm、复位 | 10 秒 |
| 受扰后的恢复 | 同中央压头动作 | 12 秒；是否恢复平静须实测 |

槽内长宽 1.6 × 0.9 m，高 0.28 m，参考水深 0.12 m。每场景 8 个固定相机。
配置为 `configs/surface_study_scene.json`，不绑定本地或服务器的粒子精度。

## 构建

Windows PowerShell，在项目根目录执行：

```powershell
& 'D:\Program Files (x86)\Blender\blender.exe' --background --python-exit-code 1 --python experiments/coupled_scenes/build_surface_study.py -- --output output/coupled_scenes/surface_study_group01_v1 --samples 32
```

支持 `--cases 01_still_water` 单独构建，以及 `--environment` 指定环境文件。
环境旁需有 `HDRI/bryanston_park_sunrise_8k.exr`。当前渲染使用本机 OptiX。
场景文件保留外部环境贴图引用；不是跨机器自包含资源包。

每个输出子目录包含可编辑 `.blend`、`colliders.usda`、`scene_spec.json`、
`cameras.json`、`motion_samples.json` 和静帧预览。动作场景另有动作峰值预览。

WSL 生成桌面总览及可离线打开的图集：

```bash
python3 experiments/coupled_scenes/review_surface_study.py --input output/coupled_scenes/surface_study_group01_v1 --output /mnt/c/Users/not_3/Desktop/液面场景_第一组
```

不启动模拟器的 USD / 动作校验（使用安装了 `usd-core` 的 Python；
Isaac 的裸 `python.bat` 不一定能直接导入 `pxr`）：

```bash
python experiments/coupled_scenes/validate_surface_study.py --input output/coupled_scenes/surface_study_group01_v1
```

## 接入流体时必须处理

- `REFERENCE_WATER_NOT_SIMULATED` 只是水位占位体；删除它，以真实模拟水面替换。晃动预览里占位体随槽刚性移动，不代表液体运动。
- 原 `pbd_pour_event` 不能直接运行这五类动作。静水与装置动作使用下述独立入口；动作通过 `--action-case` 接入，见 v5。
- 原生模拟使用 Y-up、米制。Blender 坐标转换为 `(x, -z, y)`；相机文件明确区分两种坐标系。
- `colliders.usda` 只含五块槽壁/槽底，以及需要时的一块运动装置接触形状。装饰框架、支架和水位占位体不参与水体碰撞。
- 碰撞盒厚度向外增加，保持真实槽内边界不变；接触参数仍由运行入口负责配置。
- 先预填水并排除固体占据空间，然后完成静置预热，最后把动作时间重置到零开始记录。固定 2 秒等待不能代替稳定性判定。
- 在每个求解子步调用 `coupled_scene.surface_study.motion_state`，提供运动学目标位姿；返回速度为解析速度，可用于核验或运行接口所需的速度输入，不要再对粒子施加重复推动。
- USD 动画和 JSON 表是 60 Hz 的设计采样，不能当作模拟时间步；帧 1 对应动作时间 0 秒。
- 自由传播与恢复场景共享同一扰动；有限槽有边界回波，不能将整段称为无限水域的自由传播。
- 本地可以先测静水、再测单装置。是否能以目标粒度运行长片段，应以实际显存和步耗时为准。

## 本地静水短测入口

最初增加的独立 `run_surface_study_probe.py` 仅支持静水槽（现已扩展动作）。它读取布局的五块原始
碰撞几何，初始化单个原生 PBD 水粒子集合；不加载软体，不持续加水，不加载
仅用于渲染的环境装饰。现有倾倒入口和默认配置不受影响。

```powershell
& 'Y:\isaacsim\python.bat' -u experiments/coupled_scenes/run_surface_study_probe.py --layout output/coupled_scenes/surface_study_group01_v1/01_still_water --output output/coupled_scenes/surface_study_still_probe_v1_4mm
```

默认 4 mm 间距、720 Hz、64 次迭代，目标 0.25 秒；模拟循环超过 180 秒则在
当前子步完成后停止。启动应用、初始化粒子和最终保存不计入此时限，单个
原生子步无法被该时限中断。输出目录必须不存在，避免覆盖已有结果。

这份布局生成 2,450,448 个粒子：边界留空后按间距立方计的名义水量约
156.83 L，并非把 172.8 L 几何参考体积原封不动填满。底层粒子中心离底 8 mm，
避免初始穿透；初始化会自行沉降，因此测试阶段不宣称液面已经静置稳定。

`probe_report.json` 记录子步、模拟/实际时间、求解和应用更新时间、原生速度、
穿壁数量；`final_state.npz` 保存最终原生位置与速度。它们是容量诊断，不是
已有渲染器可直接使用的生产水面缓存。可先加 `--dry-run` 无 GPU 查看规模。

### 已完成的本地首测（2026-09-08）

`surface_study_still_probe_v1_4mm` 完成全部 180 子步，即 0.25 秒模拟。
模拟循环加最后存盘约 80.73 秒，其中原生 simulate/fetch 为 72.51 秒，
app.update 为 4.81 秒；应用启动另耗约 114 秒。运行中一次 GPU 采样为
7487 MiB / 12227 MiB、68% 利用率；这不是全程峰值监测。

所有 2,450,448 个粒子保留且位置、速度有限。采样时未见侧壁/底部越界，日志
未检出 CUDA、显存不足或缓冲溢出错误。但是末态有 2011 个粒子高于槽沿，
速度 RMS 仍为 0.182 m/s。粒子中心高度的 99 分位仅为槽底以上 8.54 cm，
也不能认为当前水量已稳定达到 12 cm 参考水位。

结论仅是本地能承载本次短测，不能据此认可静水质量或长时间稳定性。
应先处理初始化/预热，再决定生产录制；本次未启动第二轮模拟。
首测原报告中的 `escaped_particle_count` 只统计侧壁/底部，不包括高于槽沿；
配套 `final_audit.json` 补充了末态高于槽沿数量。后续脚本明确拆分这两项。

## 静置后再记录

```powershell
& 'Y:\isaacsim\python.bat' -u experiments/coupled_scenes/run_surface_study_probe.py --layout output/coupled_scenes/surface_study_group01_v1/01_still_water --output output/coupled_scenes/surface_study_still_settle_v2_4mm --settle --seconds 0.5 --wall-limit 3600
```

从原始水池完整初始化，水粒子参数、阻尼、尺寸、数量与首测相同；不把速度
清零来冒充静置，不提高阻尼。默认至少预热 2 秒，最长 8 秒模拟时间，或达到
指定实际时限后停止。首次启动仍需另计应用加载时间。

`coupled_scene/surface_settling.py` 按项目约定要求连续 0.5 秒的观测窗口内：

- 全体速度 RMS ≤ 0.035 m/s，速度 99 分位 ≤ 0.09 m/s，最大速度 ≤ 0.5 m/s；
- 无侧面/底部外粒子，也无高于槽沿的粒子；
- 四个水平分区各自的粒子中心高度 99 分位变化 ≤ 2 mm。

这是可检查的就绪条件，不是通用物理标准；高度分位不是重建水面的精确水位。
不要求水面恰好恢复到 12 cm 装配参考值，也不会据此偷偷补水。

通过后保存 `settled_state.npz`，再以独立的录制时钟记录 0.5 秒、30 fps
原生位置和速度，输出到 `settled_capture/`，包含初始帧共 16 帧。
这些 NPZ 是原生数据，尚未重建/渲染水面。未通过就不生成正式记录帧；
`settled_water` 保持 false，报告指出时间上限，保留末态供检查。
预热期间每约 120 秒实际时间保存一份原生数组快照 `prewarm_checkpoint.npz`；
它不是包含求解器内部状态的精确断点文件。

CPU 就绪逻辑测试：`python experiments/coupled_scenes/test_surface_settling.py`。

### 改进：辅助预热、恢复参数、再验证（待 GPU 对照）

可选添加 `--prewarm-damping 5`：只在不录制的预热阶段临时增强原生速度阻尼，
默认保持 1.5 秒，然后用 0.5 秒平滑恢复到正式值 0.01。之后至少再运行
1 秒正式参数，且通过原有连续稳定窗口，才能保存静水并录制。辅助阻尼
阶段的数据不会进入就绪窗口；不将速度清零，不改水量、粒子间距或接触参数。
5 是待验证的实验值，不是官方推荐值；本项目 swamp 使用同类预热/恢复流程，
但其临时阻尼默认值为 0.5，不能将两种数值宣称为已验证等效。

```powershell
& 'Y:\isaacsim\python.bat' -u experiments/coupled_scenes/run_surface_study_probe.py --layout output/coupled_scenes/surface_study_group01_v1/01_still_water --output output/coupled_scenes/surface_study_still_settle_v3_assisted_4mm --settle --prewarm-damping 5 --seconds 0.5 --wall-limit 3600
```

另增加槽内速度统计，将其与全部粒子速度、槽外数量并列；分区水位统计只
使用槽内粒子。正式就绪仍然要求没有槽外/高于槽沿粒子，不会通过删水或
忽略流失粒子强行通过。v2 在 6.189 秒快照中，槽内速度 RMS 约 0.141 m/s，
槽外 2012 个粒子的速度 RMS 约 3.23 m/s；槽外统计确有影响，但不是唯一问题。

该改进不修改已启动的 v2 进程，也不同时启动第二个 GPU 测试。
辅助阻尼的端点、恢复阶段和稳定窗口已通过 CPU 单元测试；GPU 效果尚未验证。

依据：本地 `experiments/swamp_fluid/render_swamp_physx_fluid_preview.py` 的
`--settle-damping` 及恢复正式 damping 逻辑；
[NVIDIA 粒子文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html)
支持以粒子网格初始化，并说明密度约束与 rest offset/质量的关系。
因此目前没有证据要求擅自改变 4 mm 间距或随机化排列。

### v3 末态水面平滑对照

同一份 `surface_study_still_settle_v3_assisted_4mm/final_state.npz` 已分别重建
0、15、25 次网格平滑，并保持原相机、材质、灯光和 96 samples 渲染一致。
15/25 使用加权平滑；原 0 次不进行网格平滑。重建半径 0.0024814 m，
smoothing length 2.0、cube size 1.0、threshold 0.6 保持一致。
重建均输出相同数量的顶点/三角形，原始 NPZ 的 SHA256 一致。
OBJ 中的法线虽已输出，当前 Blender 导入流程由几何重算平滑着色法线，
不直接使用 OBJ 自定义法线；三个版本的导入方式一致。

桌面 `静水_v3_平滑对比` 含并排近景和同区域放大，以及六张原分辨率图片。
15 次已明显消除原先密集颗粒感；25 次进一步平滑，但此静帧中差异较小。
用户选定 25 次用于后续液面预览，`reconstruct_surface_snapshot.py` 默认改为 25；
可显式指定 0 或 15 做对照，不改已有输出文件。
本对照只验证外观，不能证明时间连续性、体积保持或物理静置门禁通过。

重建脚本支持 `--mesh-smoothing-iters 15`；对比图由
`review_surface_smoothing.py SOURCE_RUN DESKTOP_OUTPUT` 生成。

### 用户确认的就绪门槛调整

当前策略 `surface_still_v2_relaxed_speed` 按用户要求放宽速度门槛：RMS 从
0.02 调至 0.035 m/s，99 分位从 0.05 调至 0.09 m/s。最大速度 0.5 m/s、
分区水位变化 2 mm、连续 0.5 秒窗口、无槽外粒子、恢复正式阻尼后的验证
时长全部保留。这是录制就绪标准调整，不是物理参数修改或严格静止证明。
历史 v3 报告按当时门槛得到的失败状态保持原样；对旧记录回放通过新门槛，
不代表历史上已经录下视频，也不证明连续渲染不会闪动。

## v4 连续水面样片流程

`run_surface_video.py`（WSL）串联：导入 v3 `final_state.npz` → 正常阻尼下重新
验证至少 1 秒 → 录制 1 秒、30 fps 原生粒子 → 每帧 25 次加权网格平滑 →
同一低角度相机渲染 → 输出 MP4 并复制到桌面。流程状态在输出根目录的
`pipeline_status.json`，模拟和渲染分开执行，避免同时占用 GPU。

粒子初始位置、速度均原样保留，未缩放、平移、清零或补水。导入入口是
`run_surface_study_probe.py --initial-state ... --settle`，禁止同时辅助加阻尼；
导入时校验粒子数量、原生参数和槽体位置。NPZ 不包含求解器内部状态，故
必须重新验证；模拟时间从导入后 0 秒计，原快照 8 秒来源另存入报告。

录制保存 31 个快照，包含 t=1 秒终点；视频编码 t=0 至 29/30 秒的 30 帧，
得到恰好 1 秒正常速度视频。末端快照保留供检查。渲染固定随机种子、关闭
运动模糊，不做时间平滑。是否在视觉上静稳仍待观看，数值门槛通过不等于
已经验证画面。流程若未通过重新静置或录制时出现槽外粒子，不继续出片。

本轮目录：`output/coupled_scenes/surface_study_still_video_v4_4mm`。
预定桌面文件：`静水_v4_25次平滑_1秒.mp4`；只在全部步骤成功后复制。

## v5 夜间串行：第一组其余四种效果

`run_surface_overnight.py` 按顺序运行 02 晃动（10 秒）、04 推波反弹（10 秒）、
05 中央扰动恢复（12 秒），03 波纹传播共用 05 的前 6 秒，不重复模拟。
03 的整段可能包含边界回波，不能声称是无边界自由传播；05 不保证末尾完全静止。
每种效果从 v3 原生静水末态独立恢复，不继承上一场景的余波。

`--action-case` 为原生动作入口。运动学物体由每个 720 Hz 子步的解析位姿驱动；
不修改水粒子速度。每个诊断采样读取原生刚体位姿核验，偏离目标超过 0.1 mm
就报错。Blender 清除目标装置的旧动画，按原生记录的同一位姿渲染。
启动前已在无粒子小测试中核验三种运动学装置能真实移动（不等同于验证流体耦合）。

实际静置水深约 7.9 cm，因此原按 12 cm 参考水位制作的中央压头行程不足。
运行时按水体高度 99 分位将压头调到进入水面约 2 cm，实际行程约 14.1 cm；
配置原件不覆盖，修改后的动作写入每场景报告，渲染同步。推板从水上缓慢
降入工作位置（1 秒），只在录制前使用辅助阻尼及完整恢复/静置验证；不删除
与推板重叠的水粒子，不把推板直接生成在已填水中。

录制阶段可以出现物理溅水，单独报告最大槽外数量；预热就绪仍要求无槽外水。
大量速度触及上限、非有限状态、原生装置位姿不匹配会中止该场景。独立场景
继续运行，失败日志保留。使用 25 次网格平滑、1280×960、64 samples、30 fps
预览；仿真与 Blender 串行，重建阶段最多两个 CPU 工作进程。

预计目录 `output/coupled_scenes/surface_study_group01_overnight_v5`；桌面目录
`液面第一组_夜间样片_v5` 包含完成的视频、说明及实时任务状态。
原生缓存和网格保留在 Y 盘，不向 C 盘桌面复制大型中间数据。

### v5 渲染恢复（2026-09-09）

三组仿真及全部 963 帧水面重建已完成；首次串行运行的 Blender 命令遗漏
脚本参数前的 `--`，三个渲染任务均在出图前失败。启动命令已修复，历史
失败日志保留。仅补渲染时使用以下 WSL 命令，不重新模拟或重建：

```bash
python3 experiments/coupled_scenes/finish_surface_overnight.py \
  --output /mnt/y/isaacsim_work/output/coupled_scenes/surface_study_group01_overnight_v5 \
  --desktop /mnt/c/Users/not_3/Desktop/液面第一组_夜间样片_v5 \
  --layouts /mnt/y/isaacsim_work/output/coupled_scenes/surface_study_group01_v1
```

恢复前核验全部水面文件和时间序列；旧任务状态留存快照。渲染清单逐帧
保存，`render_surface_sequence.py --resume` 只复用输入与设置一致的已登记
帧；`--frame-limit 1` 可先验证首帧，清单不会将该部分结果标为完整。
补渲染串行进行，桌面状态显示实际完成帧数。MP4 经帧数、尺寸、帧率、
时长检查后复制至桌面，不覆盖已有桌面视频。

## v6 密度迭代衰减对照

`run_surface_decay_compare.py` 等 v5 渲染结束并确认 GPU 空闲后，串行执行
720 Hz × 16/32/64 次迭代，每组 4 秒。全部读取 v5 晃动的
`simulation/settled_capture/frame_0180.npz`（录制 t=6 秒，源模拟 t=7 秒），
保留位置和速度，固定槽体，不重新静置、不修改粘性/摩擦/阻尼等材质。
64 次也从同一快照重新建立求解器，避免只有实验组承担缓存恢复差异。
该快照不是完整求解器检查点，三组恢复过程一致不等于连续运行毫无差异。

`run_surface_study_probe.py --decay-test --initial-report ...` 为受限入口：
只允许指定停止时刻、匹配槽体、720 Hz、16/32/64 次迭代；既有生产快照
恢复仍禁止任意改变求解参数。测试运行中核验槽体原生位姿，记录 10 Hz
诊断指标，并保存 2 Hz 原生快照；不重建水面、不渲染整段视频。

指标按 5 cm 水柱统计 p99 水位，以内侧水柱的水位分散程度衡量大尺度
起伏；另记录分柱平均流速与柱内剩余运动、原生动能和重力势能/单位质量。
柱内剩余运动包含真实细尺度流动，不能全称噪声。平均水位推算体积仅为
变化代理，非 PhysX 原生密度，不据此单独宣称不可压缩性已得到验证。

任务状态：`output/coupled_scenes/surface_study_density_decay_v6/comparison_status.json`。
每组详细时间序列位于 `iter16`、`iter32`、`iter64` 的 `probe_report.json`。
CPU 指标测试：`python3 experiments/coupled_scenes/test_surface_decay.py`。

## v7 单变量粘性衰减对照

用户授权只把 viscosity 从 0.002 改到本机 `AddPBDMaterialWater` 在米制
场景下的预设 0.0000017；720 Hz、64 次迭代、摩擦、阻尼、涡量、表面张力、
粒子及所有碰撞设置保持不变。与 v6 相同，从晃动 t=6 秒快照续跑 4 秒，
不重新静置、不清零速度。v6 `iter64/probe_report.json` 为已有基准，不重复跑。
粘性仅通过受限 `--decay-viscosity` 参数覆盖，生产/其他测试的默认值不变。

后台入口 `run_surface_viscosity_test.py` 先校验输入并确认显卡空闲，再运行
一组模拟。结果路径 `output/coupled_scenes/surface_study_viscosity_decay_v7`。
`comparison_status.json` 自动记录两组降到 10/5/2/1/0.5 mm 的时间：统一取
10 Hz 采样，区分首次低于阈值和此后未再超过且至少观察 0.5 秒的确认时间。
记录不够长时保持未确认，不能将记录终点当成达标时刻。阈值指前述分区
水位高低差代理，不是精确波峰至波谷，也不等同于所有波动完全消失。

## v8 单变量时间步衰减对照

`run_surface_timestep_test.py` 串行跑 360 Hz、1440 Hz，各 4 秒，复用 v6
`iter64` 的 720 Hz 基准。全部保持 64 次迭代与原粘性 0.002（不继承 v7
低粘性），同一份 t=6 秒原生位置/速度、固定容器、无预热或材质改动。
`--decay-timestep-test` 是唯一允许诊断恢复改变 hz 的入口，限制为
360/720/1440 Hz，禁止同时覆盖粘性或改变迭代数。生产入口的检查不放宽。

两组输入先 dry-run 校验，显卡空闲后才启动；记录实际时间步、求解次数、
材质值、源快照哈希。完成后核对总步数为 hz×4，统一按 10 Hz 采样比较
毫米阈值衰减时间，并记录水位和局部运动，不做视频渲染。
目录：`output/coupled_scenes/surface_study_timestep_decay_v8`；状态文件
`comparison_status.json`，原生报告分别在 `hz360`、`hz1440`。

## v9 接触摩擦归零对照

复用单组运行器 `run_surface_viscosity_test.py --friction-test`，只将 PBD
水材质 friction、容器材质 static/dynamic friction 从 0.05 设为 0。
`run_surface_study_probe.py --decay-friction-test` 限定为 720 Hz、64 次迭代，
禁止同时使用粘性或时间步覆盖。保留全部碰撞体和接触距离、原粘性 0.002、
阻尼与其他参数；从同一 t=6 秒快照续跑 4 秒，无静置/清零/渲染。
它只隔离接触摩擦，不是关闭所有接触响应，也不预设为最终生产材质。

与 v6 的 iter64 基准比较相同毫米阈值的衰减时间，完成时核对水及容器
实际材质值、源哈希、初始指标、时间步、粒子数和碰撞配置。
目录：`output/coupled_scenes/surface_study_friction_decay_v9`；状态文件
`comparison_status.json`，原生记录在 `simulation/probe_report.json`。

## v11 涡量补偿连续视频

入口 `run_surface_vorticity_video.py`，从 v5 晃动停止时的同一份
`settled_capture/frame_0180.npz` 分别续跑涡量 10 和原版 0.02，各 4 秒。
v10 的 2 fps 诊断缓存不足以展示连续运动，因此两组都重新原生采样为
30 fps（121 个快照，视频使用前 120 帧）；不插帧，不复用不同恢复方式的
旧视频作为基准。`--decay-capture-fps 30` 只改变输出频率，不改变物理步长。

保留 720 Hz、64 次迭代、粘性 0.002、摩擦 0.05；25 次水面平滑，
同一相机，1280×960、64 samples、30 fps。串行完成增强版模拟/重建/渲染，
再完成原版；每个 GPU 阶段先等待空闲。输出根目录
`output/coupled_scenes/surface_study_vorticity_video_v11`，状态 `video_status.json`。
宿主机桌面 `涡量补偿_v11_4秒对比` 存放两段视频与左原版/右增强的并排视频。
此运行器要求新输出目录，不支持自动重启覆盖；中断后应先检查阶段和缓存。

## v10 涡量补偿对照

用户授权试涡量补偿。`run_surface_viscosity_test.py --vorticity-test` 单独
将 vorticityConfinement 从 0.02 提到 10（官方 SnippetPBF 示例取值），
保留 720 Hz、64 次迭代、粘性 0.002、摩擦 0.05 及所有碰撞/限速设置。
不沿用 v9 的零摩擦或 v7 的低粘性。原生入口为 `--decay-vorticity 10`，
禁止与其他诊断覆盖组合；生产默认值仍为 0.02。

从同一 t=6 秒快照续跑 4 秒，比较毫米阈值衰减时间及成片流动/柱内
剩余运动。补偿会添加运动，持续搅动不等于正确保留重力波；不到阈值
不能单独判为改善。保留越界、非有限值及大量速度触及上限的停止检查。
目录：`output/coupled_scenes/surface_study_vorticity_decay_v10`；任务状态为
`comparison_status.json`，原生记录在 `simulation/probe_report.json`。
