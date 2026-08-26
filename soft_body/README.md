# 软体核心模块

这里存放不启动 Isaac Sim、可以直接单元测试的软体公共逻辑。根目录
`soft_body_bounce_hero.py` 继续负责命令行兼容和 Isaac/PhysX 执行编排。

- `config.py`：物体配置归一化和基础字段约束。
- `geometry.py`：局部碰撞裁剪等纯几何操作。
- `tet_quality.py`：四面体体积、长宽比和相对 bind pose 的反转统计。
- `penetration.py`：精确线段/三角面相交、BVH 候选复核和穿透容差判断。

依赖方向保持单向：主入口和审核工具可以导入本包，本包不能导入
`soft_body_bounce_hero.py`，也不应导入 `omni`、`pxr` 或启动
`SimulationApp`。穿透审核把 USD/Blender 数据读取留在工具层，数值判断放在
本包，避免审核代码再次与仿真入口耦合。BVH 只负责宽阶段候选筛选，最终失败
必须由精确表面交叉和超过容差的侵入深度共同确认，正常贴合不算穿透。
Tet 拓扑按无向面的出现次数提取真实边界：出现两次的是内部共享面，不计为
非流形；反转审核只使用 simulation Tet 的可信 bind 映射。
