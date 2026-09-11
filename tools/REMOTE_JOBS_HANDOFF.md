# sglab 串行队列：本地投递说明

服务器消费者已部署。权威协议是服务器上的
`/data/jiachen/jobs/REMOTE_JOBS_HANDOFF_v1.md`；本文件只说明客户端，勿覆盖服务器协议。
任务和小型结果走 SSH/SCP；代码仍通过 Git 管理；缓存和视频按需拉取。

## 正式任务

参考服务器 `/data/jiachen/jobs/examples/001_static_water_3mm.json`，不能直接投递模板。

- `pipeline_job` 才自动执行；`agent_request` 会被标为 blocked 等待人工处理，不会自动唤起 agent。
- 必填 schema_version、唯一 id、非负整数 order、title、request、parameters、depends_on、on_failure。
- 自动任务还须填写 code、scene、execution、output、resources。
- 客户端要求真实完整 40 位提交 SHA、detached worktree；提交须已存在于服务器仓库。消费者不会 fetch/pull，也不使用现有脏工作树。
- 每步须提供 name、kind、uses_gpu 和字符串数组 argv；路径支持 `{worktree}`、`{output}`。
- parameters 只记录参数，不会自动转成命令行，实际参数须落实到 argv 或受控环境变量。
- 模式为 simulate_only 或 simulate_and_render；后者须有 simulation 和 render 步骤。
- resources.gpu_count 为 1，不能覆盖 SIM_GPU / CUDA_VISIBLE_DEVICES。
- GPU 步骤使用服务器约定的 isaac_docker_python.py / blender_single_gpu.py 包装器，不能照搬 Windows 命令。
- 输出须为新的绝对路径，例如 `/data/jiachen/job_outputs/JOB_ID`；场景资产和依赖须提前准备好。

## 本地使用

```bash
python3 tools/remote_jobs.py submit path/to/task.json --dry-run
python3 tools/remote_jobs.py submit path/to/task.json
python3 tools/remote_jobs.py status
scp sglab:/data/jiachen/jobs/results/JOB_ID/summary.md ./
scp sglab:/data/jiachen/jobs/results/JOB_ID/artifacts.json ./
```

--dry-run 仅本地初筛，不联网，不代表服务器代码、资产、空间或 GPU 检查通过。
客户端 ID 限定为字母/数字开头的 ASCII 字母、数字、下划线、连字符，最多 96 字符；服务器允许更宽范围。

正式 submit 会让任务可被执行，只用于用户已授权的具体任务：

1. SCP 到 `.incoming/<id>.<随机串>.json.part`，消费者忽略 .part。
2. 校验 SHA256，调用服务器当前 validate_job 对正文和最终文件名做校验。
3. 用原子、不覆盖的硬链接发布为 `.incoming/<id>.json`，移除本次临时文件。
4. 消费者在文件稳定后接收，管理 submitted 永久去重原件和 pending，客户端不直接写它们。

上传成功仅表示等待接收。校验失败可能留下 .part，不会执行。
SSH 中断后先查状态和同 ID 文件，勿盲目重投。重跑用新 ID，并注明替代哪个任务。

## 队列约束

- 全队列单例锁，任务和步骤串行；按 order、文件名排序。队首依赖未完成时，后面的任务不会越过它。
- GPU 连续空闲 120 秒：无计算进程、显存 ≤64 MiB、利用率 ≤5%，且无人工预约；不会停止其他人的进程。
- /data 至少保留 300 GiB 空间；这不是显存容量估算。
- stop_queue 失败时停队；continue 允许后续符合依赖要求的任务继续。
- 崩溃后 running 非空须人工核实原进程，不能自动重投。ID 去重不等于 exactly-once 执行保证。
- 状态目录为 pending、running、done、failed、blocked、invalid。
- 详细记录在 results/<id>/ 下的 status.json、events/、logs/、summary.md、artifacts.json。
- done 是执行完成，不等于物理正确或视觉验收通过。

已有 jiachen-job-queue.service 用户服务，无需启动第二个 daemon。
客户端不负责修改服务器配置、拉代码、恢复停队。部署工具不等于授权具体仿真。
