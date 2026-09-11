# 主动驱动资产包 active_drive_20260912_v1

本包仅包含已选定的外观、相机、碰撞/初始填水和图像依赖。
没有仿真缓存、视频，也不包含主动驱动应用代码；应用代码仍需单独提交同步并适配 Linux。
资产到齐不等于先前 blocked 搅拌任务已经能执行。本包关联请求 20260912_001_stirring_2x_handoff。
不可修改该请求或 submitted 去重原件；接入完成后以新 ID 建立 pipeline_job。

## 场景对应

| 场景 | designs 子目录 | simulation_assets 子目录 | 用途 |
| --- | --- | --- | --- |
| 陶瓷壶倾倒 | 01_container_transfer | 01_container_transfer | 低落差 v7 壶体 + 已认可 v10 水量，4mm |
| 玻璃碗搅拌 | 02_stirring | 02_stirring | 修补接触缝隙的 v4 外观 + v2 填水，4mm |
| 活塞推水 | 03_piston_push | 03_piston_push | v4 外观 + v2 填水，4mm |
| 活塞槽 3mm 对照 | 同上 | 03_piston_push_3mm | 同一外观，3mm 静水诊断输入；不要误用于搅拌 |

原始 assets.json 中的 source_blend Windows 路径保留为来源记录，实际路径以 manifest.json 为准。
以 assets.json 顶层和 geometry_and_fill.npz 为真实几何/填水，不要按嵌入的旧 case 尺寸重新生成。
design_spec.json 中 source_blend_sha256 可能记录派生前的设计；仿真配对校验使用 assets.json 的 source_blend_sha256。
搅拌 2 倍速在运行时传 --stirring-speed 3.6，动作窗口 --seconds 10；不要改动资产内旧动画。
视频必须使用原生缓存中的同帧装置姿态，不能直接播放 .blend 的设计动画冒充耦合运动。

## 校验及加载

资产根目录：/data/jiachen/assets/active_drive/active_drive_20260912_v1
manifest.json 固定逐文件 SHA256；外观和碰撞填水文件完全保留原字节，不重新保存 .blend。
HDRI 是从服务器已有且哈希相同的文件复制到本包的独立依赖，渲染不再依赖可变项目目录。
其余五张图像已打包在每个 .blend 内，没有链接的外部 .blend 库。

普通校验（不使用 GPU）：

```bash
python3 /data/jiachen/assets/active_drive/active_drive_20260912_v1/support/active_asset_bundle.py --root /data/jiachen/assets/active_drive/active_drive_20260912_v1
```

渲染脚本必须在 open_mainfile **之后**加入以下逻辑，否则原 Windows HDRI 路径在 Linux 无效。
路径转换仅作用于内存，不保存原始 blend，不改变任何几何、相机或灯光参数：

```python
from pathlib import Path
import sys
asset_root = Path('/data/jiachen/assets/active_drive/active_drive_20260912_v1')
sys.path.insert(0, str(asset_root / 'support'))
from active_asset_bundle import remap_loaded_images
bpy.ops.wm.open_mainfile(filepath=str(asset_root / 'designs/02_stirring/02_stirring.blend'))
remap_loaded_images(asset_root)
```

发布时使用 CPU-only Blender 实际打开全部设计，检查 HDRI 可加载、相机/运动对象存在、
NPZ 粒子数及所有哈希。验收文件为 acceptance.json；这是资产检查，不是仿真或视觉验收。
不自动投递/恢复任务，不改变运行参数，不启动 GPU，不覆盖服务器现有脏工作树。
