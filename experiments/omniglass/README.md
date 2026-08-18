# OmniGlass 实验

这些脚本记录了透明软体在 RTX OmniGlass、OmniSurface、厚体积、薄壁和折射设置下的对比实验。最终数据集已经改用 Blender Cycles 材质，因此这里不再是正式渲染入口。

各检查 wrapper 依赖同目录中的基础脚本；基础脚本会把仓库根目录加入 `sys.path`，以继续读取固定拓扑缓存模块。
