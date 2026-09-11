# 主动驱动场景：本地装置搭建

## 2026-09-12：主动驱动资产已发布到 sglab（未启动仿真）

服务器只读版本目录：
`/data/jiachen/assets/active_drive/active_drive_20260912_v1`。
原始 blend/NPZ/JSON 字节不变，manifest SHA256：
`954f5f732aca53b447254a358317c1ae6f3e2f66bc0377883c2f758b56e3249f`。

- 倾倒：lowered_design_v7 + water_minus10_assets_v10，4mm / 100434 粒子。
- 搅拌：contact_v4/02_stirring + active_02_stirring_assets_v2，4mm / 694674 粒子。
- 活塞：contact_v4/03_piston_push + active_03_piston_push_assets_v2，4mm / 696234 粒子。
- 备选活塞槽静水输入：active_03_piston_push_assets_v3_3mm，3mm / 1658712 粒子，不作为搅拌默认。

实际上传 93454180 字节；唯一外部依赖 331344990 字节 HDRI 与服务器现有文件哈希一致，
在服务器复制到包内独立路径，不重复上传，不依赖现有可变项目目录。最终资产约406MiB。
其他五张图像已内嵌，三个 blend 均没有外部链接库。
服务器 Blender 5.0.1 仅CPU打开检查：四组输入逐文件SHA256、blend/geometry配对、
NPZ粒子数、相机、运动对象和8192x4096 HDRI加载全部通过；未渲染、未仿真。
`acceptance.json` 是发布前暂存目录检查；`acceptance_final.json` 是最终路径复验。

服务器 agent 先读包内 README.md / manifest.json。渲染 open_mainfile 之后调用
包内 support/active_asset_bundle.py 的 remap_loaded_images(asset_root)，只在内存
映射 Windows HDRI 路径，不重新保存源 blend。assets.json 的 source_blend 是来源记录，
使用 manifest 的实际路径，不根据旧 case 元数据重建几何。

关联请求 `20260912_001_stirring_2x_handoff`：本次只补齐资产，未改不可变任务正文，
未自动重投、清除 blocked 或启动 GPU。主动驱动应用代码仍未提交同步，Linux重建/渲染入口
仍须接入；资产可用不等于完整搅拌流水线已可跑。
构建/检查工具：tools/build_active_asset_bundle.py、tools/audit_blend_dependencies.py、
tools/active_asset_bundle.py。外观和场景构建代码本轮没有提交到GitHub。

## 2026-09-12：v9 长暂停后同配置重跑 v9b

用户要求继续。原Windows kit PID26068与WSL队列22927仍存在，分别执行一次
NtResumeProcess/SIGCONT后，原probe触发5400秒墙钟限时（OS暂停被计入），
以 `Wall-time limit reached; partial cache retained` 退出；非流体爆炸。
保留原v9的4.2秒记录及failed_state，不冒充可恢复的完整求解器检查点。
使用同资产、同官方Water+高阻尼预热、同720Hz/64iter、同3mm参数从头重跑：
`active_piston_static_v9b_official_3mm`；完成10秒后自动渲染6–10秒。
桌面 `静水_v9b_官方Water_3mm`。旧暂停标记已注明恢复结果；不要再次恢复旧PID。

## v9：官方 Water + 高阻尼预热，3 mm

用户观看v8表示仍略抖但明显改善，要求3mm。v9复用
`active_03_piston_push_assets_v3_3mm`（1658712粒子），已检查碰撞顶点/三角
与v8逐项完全一致；填充范围/8mm初始化避障空隙相同，格点离散名义体积
比4mm多约0.51%，不能当作严格等体积。官方材质不变，所有原有距离参数
随4→3mm缩为.75：fluidRest=.0015，particle/systemContact=.0025252525，
system/solidRest=.0025，wallContact=.003、wallRest=0。
720Hz/64iter、damping5保持1.5s再.5s降0、涡量始终0、回收限制不变。
模拟10秒，完成后自动按诊断模式渲染6–10秒，不把数值或视频完成当视觉验收。
输出 `active_piston_static_v9_official_3mm`；桌面
`静水_v9_官方Water_3mm/官方Water_3mm_静水最后4秒.mp4`。
入口 `run_official_water_probe.py --assisted-prewarm --assets ...v3_3mm
--output ...v9_official_3mm --render-desktop ...`。14项CPU测试及dry-run通过。

## v8 静水完成及末4秒渲染

v8 完成10秒，126快照，696234粒子全部保留，无回收；最后静水原门槛通过，
RMS=.02559 m/s，p99=.06983 m/s。不能据此代替视觉验收。
用户要求渲染：`run_active_pour_video.py --diagnostic-static-last4` 明确走诊断
检查分支，仅选择6–10秒121原生快照。验证完整等间隔时间、装置不动、
阻尼与涡量均为0；保留 diagnostic_only 标记，不修改源manifest或冒充动作缓存。
画面沿用原相机/25次重建平滑/1280x960/64samples，30fps共120视频帧。
输出在 v8目录 `/video`；桌面 `静水_v8_官方Water_高阻尼预热/官方Water_静水最后4秒.mp4`。
渲染manifest也采用有界PermissionError重试，避免Windows读取锁中止任务。

## 2026-09-11：v8 官方 Water + 高阻尼预热

v7 未完成静水测试：0.6秒有76个满足侧向回收条件，超64个保护上限而中止，
没有执行删除；不能据此否定官方材质。用户要求恢复高阻尼预热。
v8 仅增加 `--official-water-prewarm`：前1.5秒 damping=5，随后.5秒平滑
降到官方运行值0，涡量始终0，其余v7设置及安全限制不变。只有阻尼完全恢复0
后的采样才进入静水判定，最早3秒；观察到10秒，不声称完成即验收通过。
入口：`run_official_water_probe.py --assisted-prewarm --output
/mnt/y/isaacsim_work/output/coupled_scenes/active_piston_static_v8_official_prewarm`。
GPU空闲后运行，旧v7与3mm缓存不覆盖；13项CPU测试及dry-run通过。

## 2026-09-11：v7 官方 Water 静水基线

用户同意测试本机 PhysX 110.1.13 `particleUtils.AddPBDMaterialWater` 的完整
材质预设。米制下 viscosity=1.7e-6、vorticity=0、damping=0、friction=.1、
cohesion=.01、surfaceTension=.0074。运行时直接调用该函数并逐项断言，记录
安装源码路径/hash/函数文本；不把论文 XSPH 系数或大型 SDK 落水示例当通用配方。
4 mm 的 fluidRest=.002，particleContact=systemContact=.002/(.99*.6)，
systemRest=solidRest=.002/.6；墙 contact=.004/rest=0 以及旧几何、密度1000、
720Hz/64iter、速度保险、缓冲配置不变，后者不是官方 Water 预设的一部分。

独立入口 `run_official_water_probe.py`，输出
`output/coupled_scenes/active_piston_static_v7_official_water`，资产复用
`active_03_piston_push_assets_v2`（696234 粒子）。装置固定 10 秒，不作高阻尼
预热、不升涡量；1–5 秒每秒快照、6–10 秒30fps快照，共126帧，原生 p/v/ids。
逐行沿用既有静水门槛，但 diagnostic_only/physics_gate_passed=false，完成诊断
不等于视觉合格；不会自动送入动作渲染。仅前2秒外侧下落、总计64个的原有窄
回收仍保留，超限或速度保险失败则停止。全局水默认及旧实验不改。

前序 v6 搅拌在动作2.933秒/89缓存帧处因 WinError5（进度JSON替换被拒绝）
失败，非已证实流体不稳定，未重跑。probe 的 atomic_json 对 PermissionError
新增最多7次有界重试，其他IO错误立即抛出；CPU共12项测试通过。
v7 队列等待前序终态及GPU空闲后启动，不与其竞争GPU。

## 2026-09-11：搅拌 v6 两倍转速，串行排队

用户认为搅拌不明显，要求适当加速且沿用涡量 10。本次只覆盖运行时
`speed_rad_s`：1.8→3.6（约 34.4 rpm，叶端速度约 .70 m/s）；仍为 4 mm、
694674 粒子、720 Hz、64 迭代，原几何/初始水/材质/缓存/相机不变。
0–2 秒装置不动，2–3 秒起转，3–7 秒匀速，7–8 秒停转，8–10 秒余流。
预热依旧先 .02 后恢复 10；沿用原 motion-ready 门槛，不声称静水问题已解决。

`run_active_remaining.py --cases 02_stirring --assets-version v2 --stirring-speed 3.6
--stirring-prewarm-recycling --wait-for-task .../active_piston_static_v5_3mm/status.json`
输出 `output/coupled_scenes/active_stirring_water_v6_speed2x`，桌面
`C:\Users\not_3\Desktop\搅拌_v6_两倍转速`。等 3 mm 诊断完成/失败，再等待 GPU
空闲；单独跑搅拌，完成原流程安全检查后自动重建和渲染完整 10 秒。
旧资产文件不修改；probe_report 同时保存 source_asset_motion 与实际 motion，
capture manifest 保存实际覆盖后的 motion；渲染按记录的原生姿态，不播放旧动画。

## 2026-09-11：3 mm 静水分辨率对照（v5，非成片验收）

用户观看 v3 搅拌、v4 活塞后认为静水不合格。原先的 restored 宽松门槛是
motion-ready，不是严格静水。两例在装置不动时恢复涡量 10，RMS 从约
0.02 m/s 增长到 0.067/0.078 m/s；四大区域水位统计未识别细碎局部波动。
用户要求先测试 3 mm，故只跑更浅的活塞槽，不改全局默认、不重跑动作/渲染。

入口 `run_active_resolution_probe.py`；资产 `active_03_piston_push_assets_v3_3mm`；
输出 `active_piston_static_v5_3mm`（均在 `output/coupled_scenes`）。保持真实几何、
填充包围范围和 8 mm 初始避障空隙不变，仅将格点间距 4→3 mm，粒子 offset
及墙面 contact offset 按 0.75 缩放。720 Hz、64 迭代、材质、速度上限不变；
保留原先最多 64 个、前 2 秒、仅外侧下落粒子的窄回收规则，超限仍失败。
同一填充范围的格点离散会造成少量名义体积差，比较结果时须量化。

新模式 `--stationary-resolution-probe` 固定装置 8 秒：0–3 秒涡量 .02，
3–4 秒平滑恢复到 10，4–8 秒继续观察；阻尼时间表不变。保存 3/4 秒快照及
6–8 秒每 1/30 秒原生 p/v/ids，共 63 帧，不是连续视频缓存。逐行记录原始
严格静水检查；不通过也可完成诊断，始终标记 diagnostic_only、
physics_gate_passed=false，绝不送入现有成片流程冒充合格。

对照现有 4 mm v4 的 3 秒低涡量快照、4 秒记录、6–8 秒静止片段。
CPU 回收/恢复/offset 缩放共 10 项测试通过，GPU 空闲时启动。

## 活塞后续 v4：有限预热侧向回收

接触修正版搅拌已完成 10 秒/301 帧，694,674 粒子全部保留、未触发回收，视频
继续由原队列生成。接触修正版活塞在预热 0.5 秒出现 24 个侧向逸出粒子，
直到 10 秒不再增加；最终推板后方为 0。排除它们后的 RMS=.02549、p99=.06935、
最大速度=.14038 m/s，水位变化 <.06 mm。原生仅有失败快照，不能严格证明
这 24 个是越墙而非瞬时穿墙，但不存在旧版持续向板后泄漏的数量增长特征。

用户同意有限回收。新增 `--recycle-piston-prewarm`，仅活塞案例、仅预热前
2 秒、累计最多 64 个；同时满足下降、低于槽底 2 cm、越过侧墙外界 1 cm、
位于推板最前端前方 1 cm 到远端界前 1 cm 之间才删除。板后、板内、槽底投影
内、远端之外、2 秒以后及动作阶段禁止回收。原有速度、水位和越界门槛不改。
所有阶段快照现均包含显式稳定 IDs；删除记录和原生幸存 p/v 相等检查保留。

9 项回收/恢复测试及 Windows dry-run 通过；旧失败快照的空间规则回放正好
选择 24 个（不是对前 2 秒时序的证明，实际仍待重跑核验）。单独活塞队列为
`output/coupled_scenes/active_piston_water_v4_prewarm`，桌面
`C:\Users\not_3\Desktop\活塞推水_v4_预热回收`。先等原搅拌 video_status.json
到 complete/failed，再等待 GPU 空闲，随后运行 10 秒动作并自动渲染通过结果。
不重跑、不停止或覆盖搅拌任务。实际进度以新队列 status.json 为准。

## 最新修正：active_remaining_water_v3_contactfix

v2 两项均未进入动作：搅拌有 8 个预热逃逸粒子及轮毂/桨叶缝附近局部高速；
活塞水越过推板进入后侧，再从后端掉落。只凭失败快照不能还原每个粒子路径，
但原底角的网格射线实测开口达 10.3 mm，确有结构漏水隐患。

用户授权修改后，`fix_active_apparatus_design.py` 在新目录
`active_drive_contact_v4` 派生两套可见模型，旧文件不覆盖：

- 搅拌轮毂半径 24→36 mm，和三片桨叶及轴使用精确布尔并集，去掉内部表面。
  提取网格验证为一个闭合连通实体，最大旋转半径和碗不变。
- 活塞增加与本体并集的密封边，槽底宽度 326→338 mm、倒角 12→0.5 mm，
  封住槽底与玻璃侧壁的连接空隙。推板密封边倒角 0.3 mm，底 Y=.0698、侧 Z=±.1655。
  射线复查中央至底角：底隙约 0.8 mm，侧隙约 1.5 mm，均未与槽体几何相交。
  这是几何核验，不是原生粒子密封认证，后续仿真仍需确认。

重新提取 evaluated 网格并排除固体生成 4 mm 水粒子：搅拌 694,674、活塞
696,234。填水范围未改，数量轻微下降来自固体占据空间及其接触余量变化。
新输入为 `active_02_stirring_assets_v2` / `active_03_piston_push_assets_v2`。

`--recycle-stirring-prewarm` 仅用于搅拌预热，最多累计 64 个：必须向下运动、
低于碗底 2 cm 且在碗外轮廓外扩 1 cm 之外才可删除；原生幸存 p/v 和稳定 ID
核验沿用原回收代码。超过数量上限立即停止；碗内、碗投影内穿底和动作阶段均
不回收。活塞不启用回收，不能靠删除漏水粒子冒充密封。

两套 Windows dry-run、8 项回收/恢复测试及模型闭合/连通/间隙检查通过。
队列 `active_remaining_water_v3_contactfix` 已启动：搅拌→活塞→渲染通过案例，
完整 10 秒动作，其他物理参数和相机不变。桌面目录
`C:\Users\not_3\Desktop\主动驱动_搅拌与活塞_v3_接触修正`。
当前结果以队列 status.json 和各 simulation/probe_report.json 为准，不预先声明通过。

## 最新：搅拌和活塞接入流体（active_remaining_water_v2）

陶瓷壶 v10 视频已获用户认可，用户要求继续剩余场景。复用已接受的 v3
`02_stirring` / `03_piston_push` 外观，不重建造型。新增
`prepare_active_apparatus.py` 从 evaluated Blender mesh 提取碰撞，包含可见倒角；
每个组件检查闭合及正体积，水初始化避开固体和 8 mm 接触余量。

- 搅拌：玻璃碗 + 三叶桨/轮毂/轴；695,714 个 4 mm 粒子，初始局部水顶 0.230 m。
  旋转外半径约 0.1954 m、转动件最低 Y=0.0805 m，完整旋转不会碰到碗壁/底。
- 活塞：陶瓷底、两片玻璃侧壁、远端壁 + 推板及顶边；698,775 个 4 mm 粒子，
  水仅在板前，初始局部水顶 0.210 m。底部几何间隙约 2 mm，两侧约 5.5 mm；
  间隙是否能在粒子接触距离下封水尚未实测，不能声明密封成功。
- 固定支架、电机/驱动外壳和工作台未增加碰撞；不使用隐藏水力推力，不自动
  对这两种容器启用陶瓷盆专用回收掩码。样片不是实际机械驱动/密封机构认证。

输入分别是 `active_02_stirring_assets_v1`、`active_03_piston_push_assets_v1`。
扩展现有 `run_active_pour_probe.py` 以读取原生转轴与位移，支持搅拌 Y 轴旋转、
活塞 X 轴平移；每次诊断核对原生刚体位置/四元数。仍为 720 Hz、64 次迭代、
0.02 低涡量预热→1 秒恢复 10→2 秒就绪检查；恢复后使用用户接受的 0.08/0.20
速度阈值，碰撞材质默认 0.05，不沿用陶瓷盆专用摩擦 0.30。
每个完整动作 10 秒、30 Hz 缓存 301 帧，包含动作 t=0；无粒子删除/补水。

`run_active_remaining.py` 串行跑两套物理，再串行渲染通过的案例，失败案例
保留报告而不强行出片。运行前等 GPU 空闲，单案例原生运行仍有 90 分钟墙钟
上限。当前输出 `output/coupled_scenes/active_remaining_water_v2`，桌面
`C:\Users\not_3\Desktop\主动驱动_搅拌与活塞_v2`；进度见根 status.json。
视频沿用原相机、25 次水面平滑、1280×960、64 samples、30 fps、完整 10 秒。
渲染器按 manifest 的 moving_object 和原生位姿更新物体，来源 blend 哈希必须
匹配仿真输入。旧陶瓷壶默认仍是 6 秒动作、1.5 秒起缓存、4.5 秒视频。

当前只表示两套输入和串行任务已准备/启动，不表示原生耦合或视频已经通过。
8 项回收/恢复测试、7 项主动运动测试以及脚本语法/两种配置 dry-run 已通过。
首轮 v1 在读取包含中文的 assets.json 时触发 Windows 默认 GBK 解码异常，
尚未创建物理场景；已显式指定 UTF-8，并在 Windows 端验证两套 dry-run。
随后另开 v2 队列，不覆盖 v1 失败日志或原始输入。

以下为历史记录。

## v3 陶瓷壶试水（2026-09-10，v2 重试已启动）

最新运行：`active_pour_water_v3`，桌面 `陶瓷壶倾倒试水_v3`。
v2 在 t=0.3 秒因 3 粒子短暂超过壶口而提前退出，侧/底包围范围越界仍为 0。
v3 保持 v2 的全部物理参数和 18 cm 初始填水，修正预热流程：允许初始化
瞬态在预热期间自行回落，不删除粒子、不清零速度；任何越界样本仍不允许
通过原连续就绪门槛。10 秒预热未通过仍失败，绝不带着未通过状态开始动作。
失败时额外保存原生 `failed_state.npz` 便于诊断，而不只保留统计。

首轮 `active_pour_water_v1` 在预热 t=0.2 秒中止：13 粒子高于壶口，
侧面/底部包围范围越界为 0，未进入倾倒动作。保留原始失败日志，不渲染。
第二轮仅将初始填水上限从壶底以上 24 cm 降到 18 cm，其余保持不变；
`active_pour_v2_assets` 共 73,557 粒子，名义点阵体积 4.707648 L。
新结果目录 `output/coupled_scenes/active_pour_water_v2`，桌面
`陶瓷壶倾倒试水_v2`。以下参数说明中的 111,593 粒子为第一轮历史值。

用户认可外观后授权试水。先跑陶瓷壶 → 椭圆浅盆，搅拌/活塞尚未适配。
`prepare_active_pour.py` 从已保存 v3 `.blend` 直接提取壶/盆网格，不重新用
v2 尺寸建壳、不作凸包封口；三角形分别为 3328 / 2816，检查壳体闭合且朝向一致。
几何和粒子数组在 `output/coupled_scenes/active_pour_v1_assets`，哈希与模型来源
记录在 `assets.json`。其中 `case` 保留旧设计的动作日程，**不能用其 v2 容器
尺寸或旧填水深度代替实际提取网格**；实际世界枢轴与网格以 donor/receiver 字段为准。

PhysX 支持静态/运动学三角网格：
[官方碰撞文档](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/simulation_setup/collision.html)。
本次 donor 使用原生运动学三角壳，receiver 为静态三角壳，仍须实际试水确认。
不修改可见壶/盆，不移动壶嘴，不放发射器。手柄、后支架、台面不参与本轮两容器
接触；飞出盆外的水仍保留为粒子，没有删除、回收或虚拟底板，不把它当成接住的水。

初始水由壶腔中心向实际内壁射线查询后取 4 mm 点阵，并剔除离壳小于 8 mm
的位置；上边界在壶底以上 24 cm。111,593 粒子、名义点阵体积 7.141952 L，
不是保证静置水量/水深。粒子只初始化一次，后续不发射、不删除、不覆写速度。

`run_active_pour_probe.py` 独立原生入口：720 Hz、64 次迭代、涡量 10、原材质
及限速设置，碰撞栈 512 MiB；没有软体，所以不分配服务器大型软体资源。
预热采用 5 阻尼 1.5 秒、0.5 秒恢复到 0.01，之后至少 1 秒正常参数，再按既有
0.5 秒连续就绪门槛通过才开始动作。最多预热 10 秒，整轮实际时限 90 分钟；
越出竖直壶的保守外包范围会阻止就绪；非有限状态、粒子数变化、速度触顶超过 0.5%、
原生枢轴/旋转与目标不符均中止。包围范围检查并非精确壳内穿透深度审计。

动作时间从 0 开始，保持原 2–5 秒倾倒、5–7 秒停留日程，本次只模拟到 6 秒。
捕获动作 t=1.5–6 秒、30 fps、136 个原生快照；视频使用前 135 帧共 4.5 秒。
渲染清除旧动画，按原生世界位置/四元数更新壶及其子物体，水面 25 次平滑。
`run_active_pour_video.py` 自动串行模拟、重建、渲染，失败不强行出片。
结果 `output/coupled_scenes/active_pour_water_v1/video_status.json`；桌面
`陶瓷壶倾倒试水_v1`。这是一轮观察性短测，不宣称壶口流量、盆外损失或物理
保真已经得到验证。其余新装置仍维持外观样机状态。

## 当前外观审阅：v3（未适配物理）

用户认为 v2 方槽/台架过于笨重，现另做外观样机，不覆盖 v2 的布局与碰撞资产。
`build_active_drive_design.py` 生成真实 Blender 模型、10 秒装置动作、8 个固定
相机及预览；没有生成水、没有输出碰撞 USD。新造型不兼容旧 v2 碰撞体。

- 倾倒：有真实壁厚/开放杯口与壶嘴的陶瓷壶，配低矮椭圆陶瓷盆，单侧后置支撑。
- 搅拌：圆形薄壁玻璃容器，细轴与小型三叶桨，紧凑上置电机与单侧后支撑。
- 活塞：长观察槽、端部集成外壳、底座内收纳驱动概念，无外露长推杆；
  底部滑台/密封/槽口及碰撞结构仍待设计，不把外观壳体当成已验证机械实现。

输出 `output/coupled_scenes/active_drive_design_v3`，桌面 `主动驱动外观_v3`。
`review_active_drive_design.py --input ... --previous output/coupled_scenes/active_drive_group_v2
--output ...` 生成新旧对照与可编辑场景图集。新样机不放水位占位块，避免干扰外观审阅。
相机对装置运动范围作 2 Hz 布局采样取景检查；不等于多视角环境遮挡/物理碰撞验证。
认可外观后再为这些新几何制作配套碰撞和运行适配，不启动长仿真。

## v2 布局记录（外观已被要求重做）

当前为布局/动作阶段，**没有运行流体**。三套装置均放进现有 warehouse 环境，
不覆盖原 `.blend`，每套保留 8 个固定相机与 10 秒装置动作。
最终布局输出 `output/coupled_scenes/active_drive_group_v2`；v1 保留为装配修正前版本。

| 场景 | 装置尺寸（长 × 高 × 宽） | 动作 |
| --- | --- | --- |
| 两容器倾倒 | 接水槽 1.30 × 0.32 × 0.75 m；上方容器 0.44 × 0.30 × 0.32 m | 0–2 s 等待，2–5 s 绕固定出水沿转至 105°，保持到 7 s，9 s 回正 |
| 搅拌 | 水槽 0.80 × 0.38 × 0.80 m；桨叶跨度 0.46 m | 2–3 s 加速，3–7 s 保持 1.8 rad/s，7–8 s 减速，8–10 s 停机 |
| 活塞推水 | 水槽 1.50 × 0.34 × 0.45 m | 2–5 s 平滑推进 0.45 m，随后保持，不回抽 |

倾倒使用**上方容器预填水、下方空槽**，不使用发射器冒充两容器间的水量转移。
搅拌器预先位于水中，初始化必须排除桨叶/轴占据空间。
活塞只在板前预填水，板后是干区；几何底部/侧面留 4 mm 间隙，尚未验证
对应 PBD 接触距离下的漏水和密封效果，不声称是完全密封活塞。
倾倒支架移到接水槽外；活塞采用贯通式长推杆并有板顶连接块，整个行程保持连接。

## 文件与生成

- 配置：`configs/active_drive_scene.json`。
- 几何与解析运动：`coupled_scene/active_drive.py`，独立于 Blender/PhysX。
- 构建器：`build_active_drive.py`，Blender 5.0.1 验证。
- 校验器：`validate_active_drive.py`，只需 `usd-core`，不启动 GPU。
- 桌面图集：`review_active_drive.py`，复用已经渲染的 PNG。

```powershell
& 'D:\Program Files (x86)\Blender\blender.exe' --background --python-exit-code 1 --python experiments/coupled_scenes/build_active_drive.py -- --output output/coupled_scenes/active_drive_group_v2 --samples 32
```

构建只接受不存在的输出目录。每个子目录包含可编辑 `.blend`、`colliders.usda`、
`scene_spec.json`、`motion_samples.json`、`cameras.json`、布局检查与四张预览图。
所有 `.blend` 保持外部环境/HDRI 引用，不是跨机器自包含资产包。

```bash
python experiments/coupled_scenes/validate_active_drive.py --input output/coupled_scenes/active_drive_group_v2
python -m unittest discover -s tests -p test_active_drive.py
python experiments/coupled_scenes/review_active_drive.py --input output/coupled_scenes/active_drive_group_v2 --output /mnt/c/Users/not_3/Desktop/主动驱动场景_v2
```

## 坐标、运动和已检查范围

米制，Isaac Y-up，Blender 转换 `(x,y,z) → (x,-z,y)`。
`body_pose(body, motion, seconds, origin)` 返回世界位置、转轴、弧度角及解析
线速度/角速度。旋转绕刚体根节点：倾倒根节点固定在出水沿，搅拌根节点在轴心。
时间轴/相机统一帧 1 = t 0；USD 与 Blender 为 60 Hz 设计采样，不代表求解步长。

检查包括解析速度与位置导数、动作端点、倾倒枢轴固定、活塞不回抽、板前水区、
理想体积几何余量、整个行程中运动接触形状不与静态槽壁相交（301 个采样时刻；
圆柱用保守包围盒）、601 帧 USD 变换与解析定义一致、Blender 平移及四元数一致。
8 个相机根据装置扫过范围在搭建阶段确定位置，随后固定不动。视锥覆盖检查不
代表所有视角都没有环境遮挡，也不是多视角图像质量认证。

`REFERENCE_FILL_NOT_SIMULATED` 仅初始帧显示；从帧 2 隐藏，避免将刚性水块随
容器转动冒充真实流体。动作预览中没有水，不是模拟水被删除。该占位物不写入碰撞 USD。
槽壁、桨叶与水中转轴、活塞板为物理形状；工作台、外置支架、驱动外壳和板顶
推杆是装饰，不参与流体碰撞。预填参考体积不是实际粒子水量或沉降后的实际水深。

## 下一步接入流体（尚未实现）

旧 `pbd_pour_event` 和 `run_surface_study_probe.py --action-case` **不能直接运行
这三套布局**。需要专用适配器，读取布局并在每个原生子步驱动运动学位姿，
保留水与移动容器/桨叶/推板的 PhysX 接触响应，不给水粒子额外施加重复推动。

初始化排除固体占据空间，静置到录制就绪后将动作时钟归零。复核实际沉降水位
是否覆盖桨叶，活塞剩余空间是否足够、间隙是否漏水。若调整装置位置/行程，应
同步更新原生记录和渲染，不让碰撞体与可见装置错位。
初始候选水参数保持 4 mm、720 Hz、64 次迭代、涡量 10；这些参数在别的短测
中可用，不等于本组三种耦合已验证。资源容量由运行端按场景设置，不套服务器
显存预设到本地。未授权前不启动长时间流体仿真。

## 陶瓷壶静置单变量对照 v4

`run_active_pour_probe.py` 支持 `--settling-only --settling-vorticity 0.02`。
此组合只运行完整 10 秒直立静置，不提前转入倾倒，不渲染，不改变全局涡量默认值。
使用 `active_pour_v2_assets` 的同一初始位置、零初速度和精确壶体网格，与
`active_pour_water_v3/simulation` 的涡量 10 静置记录对照。4 mm、73,557 粒子、
720 Hz、64 次迭代、阻尼释放时序及原静置门槛均不变。

输出：`output/coupled_scenes/active_pour_settling_v4_low_vorticity`。
报告记录首次通过时间、每次采样是否通过及 10 秒末是否通过；只有末尾仍通过
才标记 completed。保存 `settling_final_state.npz`，失败时另存失败快照。
这不是低涡量静置后恢复 10 的测试；恢复策略尚未实现或验证。

实测完成：仿真 10 秒，循环墙钟约 194.7 秒。3.0 秒首次通过，之后所有 10 Hz
采样至 10 秒均通过未修改的门槛。末帧 RMS 0.02604 m/s、p99 0.06943 m/s、
最大速度 0.14766 m/s；原涡量 10 对照分别为 0.06801、0.16457、0.27967 m/s。
末两秒四区水位最大时间变化 0.215 mm（原对照 0.770 mm）。粒子仍为 73,557，
末帧保守侧/底越界及越壶沿计数均为 0；这不是精确网格穿透深度认证。
此结果支持降低静置涡量，但不证明正式动作前恢复 10 后仍能维持静水。

```powershell
& 'Y:\isaacsim\python.bat' experiments/coupled_scenes/run_active_pour_probe.py --assets output/coupled_scenes/active_pour_v2_assets --output output/coupled_scenes/active_pour_settling_v4_low_vorticity --settling-only --settling-vorticity 0.02
```

## 陶瓷壶静置后恢复涡量 v5

`run_active_pour_probe.py --restore-vorticity-after-settling` 从相同初始填充开始，
以 0.02 静置。原门槛通过后，按每个原生子步在 1 秒内 smoothstep 恢复到默认
涡量 10，再以正常阻尼 0.01 保持壶直立观察 2 秒。只有该完整 2 秒窗口通过
原速度、水位及保守越界阈值，才开始原定 6 秒动作和 136 帧原生缓存。
恢复检查失败则保存快照并停止，不进入倾倒或自动渲染。

输出：`output/coupled_scenes/active_pour_water_v5_restore`。报告每行记录实际
USD 材质涡量值与阶段；保存低涡量就绪和恢复后的原生 p/v 快照。全局默认参数
不变。恢复试验不能与 `--settling-only` 或 `--settling-vorticity` 同时使用。
CPU 检查：`python -m unittest discover -s experiments/coupled_scenes -p test_active_pour_restoration.py`。

v5 实测：低涡量阶段在 3.0 秒通过，4.0 秒恢复到 10，6.0 秒恢复检查失败并
自动停止，倾倒未开始、捕获帧数为 0。3 秒 RMS/p99 为 0.02223/0.05512 m/s，
6 秒升至 0.06901/0.16542 m/s，接近从一开始使用 10 的 v3 稳态速度。
因此低涡量预热再平滑恢复 10 并没有解决此壶的静置速度超标；不能把过门槛
时刻前移来冒充在正式参数下静置通过。低涡量与恢复后快照均已保留。

```powershell
& 'Y:\isaacsim\python.bat' experiments/coupled_scenes/run_active_pour_probe.py --assets output/coupled_scenes/active_pour_v2_assets --output output/coupled_scenes/active_pour_water_v5_restore --restore-vorticity-after-settling
```

## 陶瓷壶恢复后放宽速度就绪门槛 v6

用户明确接受目前的小尺度运动，要求放宽门槛。新增显式选项
`--relaxed-restored-readiness`，必须与 `--restore-vorticity-after-settling` 联用。
仅恢复到 10 后的两秒检查改为 RMS <= 0.08 m/s、p99 <= 0.20 m/s；最大速度
0.5 m/s、区域水位时间变化 2 mm、保守越界限制不变。低涡量初始静置与其他
场景的 `SettlingGate` 默认值不变。报告明确标记 `pitcher_motion_ready_v1_relaxed_restored_speed`，
这是接受小尺度运动后的动作就绪判定，不是严格静水认证。

输出 `output/coupled_scenes/active_pour_water_v6_relaxed`；通过后继续原定 6 秒
倾倒动作，生成 136 帧原生缓存。本命令不自动渲染。

```powershell
& 'Y:\isaacsim\python.bat' experiments/coupled_scenes/run_active_pour_probe.py --assets output/coupled_scenes/active_pour_v2_assets --output output/coupled_scenes/active_pour_water_v6_relaxed --restore-vorticity-after-settling --relaxed-restored-readiness
```

## 降低倾倒落差 v7（布局与物理输入已生成，短测已启动）

用户选择降低落差。`lower_active_pour_design.py` 从已接受的 v3 陶瓷布局派生
新文件，将壶根节点及其完整动画下移 0.20 m，并缩短固定后支架、下移转轴。
壶把手与活动托架随父节点移动；盆、工作台、相机、壶/盆网格和动作时序不变。
原版布局及 v6 缓存均保留。工作台仍不参与碰撞，本次没有顺带增加碰撞体。

- 设计：`output/coupled_scenes/active_pour_lowered_design_v7/01_container_transfer.blend`。
- 原生输入：`output/coupled_scenes/active_pour_lowered_assets_v7`。
- 壶轴世界 Y 从 0.899997 改到 0.699997 m；盆沿最高 Y 为 0.369997 m。
- 对每个壶体顶点解析检查完整 0 到 -105 度转角范围，最低壶体与最高盆沿
  仍有 0.027 m 竖直余量；此分离界保证两个容器网格全程不相交。
- 重提取后的壶/盆局部顶点及三角形与旧输入逐元素相同；73,557 个 4 mm 粒子
  全部仅下移 0.20 m（浮点误差 < 1e-6 m），初速度、水量和局部填充形状不变。
- `assets.json` 顶层 donor/receiver 与 NPZ 是当前精确几何；嵌入 case 的旧 v2
  size/depth 字段不是这版陶瓷容器尺寸，不应拿来推导实际碰撞或水量。

用户随后明确要求开跑，已用以下命令启动短测；结果以输出目录的
`probe_report.json` 为准，启动不代表仿真通过。开跑前 GPU 为 114 MiB、0%。

```powershell
& 'Y:\isaacsim\python.bat' experiments/coupled_scenes/run_active_pour_probe.py --assets output/coupled_scenes/active_pour_lowered_assets_v7 --output output/coupled_scenes/active_pour_water_v7_lowered --restore-vorticity-after-settling --relaxed-restored-readiness
```

渲染时必须使用新的 `active_pour_lowered_design_v7` blend，不能沿用旧高度的
固定支架布局。降低落差只减少重力加速距离，不承诺完全消除溅出或限速停止。

## 溅出粒子原生回收 v8

用户明确要求回收溅出粒子。v8 沿用 v7 降低 20 cm 的几何及 v6 放宽后的
恢复就绪门槛，仅增加 `--recycle-escaped`。在倾倒阶段读取原生状态后、执行
速度保护和缓存之前，删除同时符合下列条件的粒子：

- 世界 Y 低于接水盆最低顶点 0.02 m；
- 位于盆外接椭圆轮廓（X/Z 半径各外扩 0.01 m）之外；
- 原生 Y 速度为负，正在下落。

此规则专用于当前轴对齐、居中的椭圆盆。盆内高速水、盆上空飞溅、上升粒子
以及在盆投影内疑似穿底的粒子均不回收，仍受原速度保护检查。未启用新发射，
不将水移回壶或盆，也不是渲染裁剪。物理输入仍为 73,557 粒子，活动数允许下降。

沿用已有 `ConditionedInlet` 的原生 USD 数组缩减方式，同步缩减 positions、
velocities、protoIndices，保持密度派生的单粒子质量。每次更新立即检查所有
幸存粒子的 p/v 与更新前逐元素相同。应用级稳定 ID 随压缩保存到每帧 NPZ 的
`ids`，不能再把帧内行号当作跨帧 ID。每次删除的 ID、位置、速度及时间写入
`escaped_removals.jsonl`；报告记录累计删除数，原始总数=活动数+删除数。

输出：`output/coupled_scenes/active_pour_water_v8_recycle`，短测已启动，实际结果
以 `probe_report.json` 为准；不会自动渲染。v6/v7 原始失败缓存完整保留。

```powershell
& 'Y:\isaacsim\python.bat' experiments/coupled_scenes/run_active_pour_probe.py --assets output/coupled_scenes/active_pour_lowered_assets_v7 --output output/coupled_scenes/active_pour_water_v8_recycle --restore-vorticity-after-settling --relaxed-restored-readiness --recycle-escaped
```

CPU 回收和恢复相关 8 个测试通过。对 v7 失败快照做只读规则回放：6,643 个
粒子符合回收条件，剩余粒子的限速比例约 0.003%，低于原 0.5% 停止阈值。
这是回放检查，不代表 v8 原生删除已经验证或倾倒已完成。

v8 后续实测完成：6 秒动作全部跑完，136/136 帧缓存完整。原生回收 19 次，
累计删除 6,957 粒子，剩余 66,600，合计仍为初始 73,557。逐次删除后的原生
幸存 p/v 相等检查全部通过。离线核验每个删除事件都符合位置/方向规则、无
重复删除，并逐帧确认 136 个 NPZ 的显式 ID 恰好等于该时刻应保留的原始 ID。
没有靠放宽速度保护跑完。此结果是仿真与回收数据通过，尚未渲染或视觉验收。

### v8 视频渲染任务

用户已要求渲染，任务输出 `output/coupled_scenes/active_pour_video_v8_recycle`，
桌面目录 `C:\Users\not_3\Desktop\陶瓷壶倾倒_v8_低落差回收`。
`run_active_pour_video.py --simulation` 复用已完成的 v8 原生缓存，绝不重新仿真。
启动前核对报告/输入/blend 的几何来源哈希，渲染器再次检查新布局 blend 哈希。
136 个快照按 25 次平滑重建水面，2 个 CPU 重建任务并行；GPU 空闲后使用
降低后的 v7 陶瓷布局、原相机、1280×960、64 samples、30 fps 渲染。
最终编码 135 帧 4.5 秒视频（动作 1.5 秒起，含倾倒前短暂等待），文件名
`陶瓷壶倾倒_v8_低落差回收.mp4`，完成后复制到桌面目录。
进度以 `video_status.json` 或桌面 `任务状态.json` 为准；此条记录不代表渲染完成。

v8 视频后续已完成：桌面 MP4 已通过 1280×960、30 fps、135 帧、4.5 秒核验。

## 增加水量和接水盆摩擦 v9

用户要求增加水量，同时认为盆底太滑。保持降低后的 v7 布局，重新生成
`active_pour_more_water_assets_v9`，填充上限从 0.18 改为 0.24 m；粒子从 73,557
增加到 111,593（+51.71%），仍为 4 mm。壶/盆网格逐元素相同，初速度仍为零。
此高度是初始化填充范围，不是静置后水深；nominal_lattice_liters 也不当作实测体积。

原生 probe 新增 `--receiver-friction`，默认仍为 0.05。本轮选 0.30，为整个
接水盆壳体（底和侧壁）设置独立的刚体物理材质，static/dynamic 同值，restitution=0。
壶壁摩擦 0.05、水 PBD 摩擦 0.05、粘度 0.002、正式涡量 10 均不变。
这是候选效果设置，不声称是标定过的水/陶瓷摩擦；报告区分 authored 材质系数
与未测量的有效接触摩擦。每个壳体的 physics 材质绑定在运行时核验。
由于本轮同时增加水量和盆材质摩擦，不能把结果变化单独归因于摩擦。

v9 短测已启动，输出 `output/coupled_scenes/active_pour_water_v9_more_friction`。
继续采用低涡量预热、恢复至 10、放宽的动作就绪速度门槛、盆外下落粒子原生回收。
不会自动渲染。旧 v8 仿真及视频保留。

```powershell
& 'Y:\isaacsim\python.bat' experiments/coupled_scenes/run_active_pour_probe.py --assets output/coupled_scenes/active_pour_more_water_assets_v9 --output output/coupled_scenes/active_pour_water_v9_more_friction --restore-vorticity-after-settling --relaxed-restored-readiness --recycle-escaped --receiver-friction 0.30
```

## v10：按用户要求将 v9 水量减少 10%

v9 停在 10 秒低涡量预热：26 粒子越出静置范围，未进入倾倒。用户要求减少
10% 水量，不改门槛或扩大预热回收。v10 从 111,593 减为 100,434 粒子（整数
四舍五入），初始零速度、4 mm 间距、盆摩擦 0.30 以及其他设置不变。
`prepare_active_pour.py --particle-limit 100434 --fill-height 0.24` 按高度保留
完整下层，最顶层只保留中心连续片区，不随机稀疏整个水体。新初始最高粒子
位于几何壶底上方约 0.222 m。已验证新点集是旧点集子集、顶层以下逐点相同，
壶和盆局部网格逐元素相同。

输入 `active_pour_water_minus10_assets_v10`；输出 `active_pour_water_v10_minus10`，
短测已启动，不自动渲染，实际状态以 probe_report.json 为准。

```powershell
& 'Y:\isaacsim\python.bat' experiments/coupled_scenes/run_active_pour_probe.py --assets output/coupled_scenes/active_pour_water_minus10_assets_v10 --output output/coupled_scenes/active_pour_water_v10_minus10 --restore-vorticity-after-settling --relaxed-restored-readiness --recycle-escaped --receiver-friction 0.30
```

v10 实测完成：预热在 6 秒通过并进入原 6 秒动作，136/136 帧完整。初始
100,434 粒子，累计回收 17,712，最终保留 82,722；未触发速度保护。用户随后
要求渲染，使用已有原生缓存，不重新仿真。渲染输出
`output/coupled_scenes/active_pour_video_v10_more_water`，桌面目录
`C:\Users\not_3\Desktop\陶瓷壶倾倒_v10_增水增摩擦`。
沿用 v8 的降低后布局、固定相机、25 次水面平滑、1280×960、64 samples、
30 fps 和 4.5 秒编码范围。旧 v8 视频不覆盖。渲染进度以 video_status.json
为准，最终视频名为 `陶瓷壶倾倒_v10_增水增摩擦.mp4`。
