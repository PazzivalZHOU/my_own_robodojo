# camera_delay 实验

人工 RGB 相机帧延迟（artificial camera frame delay）实验。

目标：在**不改仿真时序**的前提下，只延迟送给策略的 RGB 相机观测若干帧，
研究视觉陈旧（visual staleness）对 RoboDojo 评估成功率/分数的影响。

第一轮支持 `delay_frames ∈ {0, 1, 2, 4}`；`0` 为无扰动基线。

---

## 1. 目录结构

```
experiments/camera_delay/
  README.md          # 本文件
  SERVER_RUNBOOK.md  # 服务器部署/同步/回滚操作手册
  run_eval.py        # 双模式：driver（默认） / sim-client entry（被 eval_policy.sh 调用）
  perturbation.py    # FIFO 延迟缓冲 + 可选的 policy-input 录像；不依赖 Isaac
  analyze.py         # 结果聚合：analysis.md / analysis.csv / delay_vs_performance.png
  config.py          # DelayConfig / RunConfig + YAML 加载
  configs/
    base.yaml        # 评估/运行参数（task、ckpt、conda env…）
    delays.yaml      # 延迟扫描列表 + 扰动选项
  tests/
    test_camera_delay.py  # 不依赖 Isaac/GPU 的最小回归测试
  results/           # 运行产物（首次运行时自动创建）
```

三个子模块（`XPolicyLab`、`third_party/IsaacLab`、`third_party/curobo`）**零改动**。

---

## 2. 拦截点与原理

**拦截 `EvalEnv.get_obs_batch`**（`src/eval_client/eval_env.py:269`）——策略观测的唯一出口：

- `EvalEnv.get_obs()` 内部即 `self.get_obs_batch(env_idx_list=[0])[0]`；单体/批量部署都经过它。
- 该方法内已对每个 env 做 `deepcopy`，之后 `vision` 与 `state`/`action` 是并列字段，
  因此只替换 `env_data["vision"][camera]["color"]`，不延迟本体感知、depth 或相机参数。
- 延迟用每个 `env_idx × camera` 的 `collections.deque(maxlen=delay_frames + 1)` 实现，
  属于纯数据层，不改物理步进/渲染。

实现方式（**不改任何 Python 核心代码**）：
在 sim-client 进程里，`run_eval.py` 导入官方 `src.eval_client.main`，
包装其 `create_eval_env`，拿到 `EvalEnv` 实例后由 `perturbation.install(env, cfg)`：

```
env.get_obs_batch = wrapped(orig)   # 产出后，用延迟帧替换 color；last_frame 放行
env.reset         = wrapped(orig)   # 每个 episode 开始前清空缓冲区
```

### 延迟语义

- `delay_frames = d`：第 `t` 次观测返回第 `t - d` 次采集的帧。
- 预热（不足 `d + 1` 帧）采用 `warmup_mode: repeat_first`：重复最早采集的那一帧。
- `delay_frames = 0`：**根本不安装包装**，实例方法保持官方原样。

---

## 3. 如何运行

```bash
cd <RoboDojo 根目录>

# 先编辑 configs/base.yaml 里的 task / ckpt / policy_env，或命令行覆盖：
python experiments/camera_delay/run_eval.py \
  --task stack_blocks \
  --ckpt <CHECKPOINT_NAME> \
  --policy-env RoboDojo \
  --eval-num 1 \
  --delays 0,1,2,4
```

driver 会对每个 delay 值：
1. 对启用扰动的档位写 `results/<时间戳>/delay_<d>/client_config.json`；
2. 对 `d>0` 设置 `ROBODOJO_CLIENT_ENTRY` / `ROBODOJO_CAMERA_DELAY_*`；`d=0`
   且不录制 policy input 时直接使用官方客户端入口；
3. 调用官方 `bash scripts/robodojo.sh eval …`（server + client 全流程复用）；
4. 校验子进程返回码、`_result.json`、`success_rate`、`score` 和
   `eval_time >= eval_num`，再归档并写 `summary.json`。任一档不完整时 driver 返回非零，
   并将可用的半截结果保存为 `_result.partial.json`，避免被分析脚本误计入。

调试命令预览（不真正运行）：

```bash
python experiments/camera_delay/run_eval.py --task stack_blocks --ckpt X --dry-run
```

### 可选：记录“策略实际看到的（延迟）帧”

默认关闭，且**不影响** episode 的 ground-truth 录像（官方 `_stream_vision` 记录的是当前真实帧）。
开启后会把延迟帧额外存成 `delay_<d>/policy_input/env<N>_ep<M>_<cam>.mp4`：

```bash
python experiments/camera_delay/run_eval.py --task stack_blocks --ckpt X --record-policy-input
```

---

## 4. 配置字段

`configs/base.yaml`（→ `RunConfig`）：`dataset, policy_dir, task, ckpt, env_cfg,
action_type, seed, eval_num, policy_env, eval_env, policy_gpu, env_gpu,
robodojo_sh, results_dir`。

`configs/delays.yaml`（→ `RunConfig` + `DelayConfig`）：`delays, warmup_mode,
cameras, record_policy_input`。

`DelayConfig` 字段：`delay_frames, warmup_mode, cameras, record_policy_input,
policy_input_dir, record_fps`。

---

## 5. 基线保证（delay_frames = 0）

两层保证：

1. **不设置实验入口**：driver 在 `d=0` 且不录制 policy input 时会清除
   `ROBODOJO_CAMERA_DELAY_*` / `ROBODOJO_CLIENT_ENTRY`；`scripts/eval_policy.sh` 的
   `${ROBODOJO_CLIENT_ENTRY:-src/eval_client/main.py}` 展开为原值，跑的完全是官方入口。
2. **走实验入口但 `delay_frames == 0`**：`perturbation.install()` 检测到 `active == False`
   后**不包装任何方法**、不建队列、不做额外 `deepcopy`，观测代码路径与基线一致。

因此 `d=0` 可直接作为无扰动对照。

---

## 6. 上游改动记录（Upstream change record）

本实验对 RoboDojo 核心代码**只做了一处、一行**修改，用于让实验入口替换仿真客户端脚本，
且默认行为不变。

**文件**：`scripts/eval_policy.sh`
**位置**：原第 146 行的 `python` 调用处

```diff
   set +e
-  python -u src/eval_client/main.py \
+  # ROBODOJO_CLIENT_ENTRY is a generic, opt-in override for the sim-client
+  # entrypoint. It is used by experiments/camera_delay/ (see that README's
+  # "Upstream change record"); when unset it expands to the stock entrypoint,
+  # so baseline behavior is byte-for-byte unchanged.
+  python -u "${ROBODOJO_CLIENT_ENTRY:-src/eval_client/main.py}" \
```

说明：
- 这是整个 RoboDojo 中**唯一**硬编码 `src/eval_client/main.py` 的可执行位置
  （`scripts/README.md`、`CLAUDE.md` 只是文档；`XPolicyLab/policy/starVLA/...`
  属于子模块，未改动）。
- 该改动是**通用**的入口覆盖机制，不包含任何相机延迟语义；实验逻辑全部在 `experiments/camera_delay/`。
- **回滚**：删除上面这段注释并还原那一行为 `python -u src/eval_client/main.py \` 即可。

**文件**：`.gitignore`
**改动**：新增忽略 `.DS_Store` 与 `experiments/camera_delay/results/`，避免误提交实验产物（mp4/指标）和 macOS 垃圾文件。
**回滚**：删除对应的两段新增行即可。

其余文件一律未改；三个子模块（`XPolicyLab`、`third_party/IsaacLab`、`third_party/curobo`）零改动。

---

## 7. 限制与注意事项

- 延迟单位是**高层观测步**（默认配置约 25 Hz），不是物理步（250 Hz），也不是现实秒。
- 每个相机只替换 `color` RGB 字段；depth、内外参、shape 等始终来自当前观测。
- 延迟线状态在 `env.reset()` 时清空；`last_frame=True` 的收尾观测不延迟、不推进缓冲。
- 默认录像为 ground-truth 当前帧；需要核对延迟输入时用 `--record-policy-input`。
- 批量模式：缓冲按 `env_idx` 独立维护；ACT 的 `deploy.yml` 默认 `eval_batch: false`
  （`main.py` 会强制 `num_envs=1`），但代码支持批量。
- 多 GPU/多进程并发：各进程延迟状态互不影响；结果按 run 归档。

---

## 8. macOS 本地构建验证（不启动 Isaac）

本地只负责代码构建与 dry-run；真实实验只在 Linux GPU 服务器上执行。在含
PyYAML/NumPy 的 Python 环境中：

```bash
python -m unittest discover -s experiments/camera_delay/tests -v
python experiments/camera_delay/run_eval.py \
  --task stack_blocks --ckpt PLACEHOLDER --policy-env RoboDojo --delays 0,1 --dry-run
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/ACT --task stack_blocks --ckpt PLACEHOLDER \
  --policy-env RoboDojo --eval-env RoboDojo --eval-num 1 --dry-run
```

dry-run 只验证配置和命令链，不能代替服务器上的单回合冒烟。

---

## 9. 结果分析

`analyze.py` 读取各档 `delay_<d>/_result.json`（或退回读 `summary.json`），自动跳过
`complete=false` 的运行，并生成对比表与图：

```bash
# 最新一次 sweep
python experiments/camera_delay/analyze.py

# 指定某次 sweep
python experiments/camera_delay/analyze.py \
  --results experiments/camera_delay/results/<时间戳>

# 合并多次 sweep（例如多个 seed / 布局）→ 每档给出 均值 ± 标准差
python experiments/camera_delay/analyze.py \
  --results experiments/camera_delay/results/<stampA> experiments/camera_delay/results/<stampB>
```

输出（默认写到该 sweep 目录，或在合并时写到 `results/_analysis_<时间戳>/`）：

```text
analysis.md                  # Markdown 对比表（含相对 d=0 的 Δ）
analysis.csv                 # 同数据，CSV
delay_vs_performance.png     # success_rate / score 随 delay 变化（需 matplotlib）
```

示例表：

```
| delay | runs | success_rate | score       | eval_time | Δ success | Δ score |
|------:|-----:|-------------:|------------:|----------:|----------:|--------:|
| 0     | 2    | 0.650 ± 0.050| 72.5 ± 2.5  | 20        | +0.000    | +0.0    |
| 2     | 2    | 0.450 ± 0.050| 50.0 ± 5.0  | 20        | -0.200    | -22.5   |
```

若只想看表不要图，加 `--no-plot`；matplotlib 缺失时脚本会自动跳过画图并照常输出表格。
