# 软体核心模块

这里存放不启动 Isaac Sim、可以直接单元测试的软体公共逻辑。根目录
`soft_body_bounce_hero.py` 继续负责命令行兼容和 Isaac/PhysX 执行编排。

- `config.py`：物体配置归一化和基础字段约束。
- `geometry.py`：局部碰撞裁剪等纯几何操作。
- `tet_quality.py`：四面体体积、长宽比和相对 bind pose 的反转统计。

依赖方向保持单向：主入口和审核工具可以导入本包，本包不能导入
`soft_body_bounce_hero.py`，也不应导入 `omni`、`pxr` 或启动
`SimulationApp`。后续穿透审核应把 USD/Blender 数据读取留在工具层，数值判断
放在本包，避免审核代码再次与仿真入口耦合。
