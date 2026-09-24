# SERVER_RUNBOOK — camera_delay 实验（Linux GPU 服务器）

> 约定：**agent 只在本地个人电脑工作；服务器只执行本文件列出的白名单命令。**
> 个人目录（导师分配）：`/home/ypwen/zyh/`

---

## 0. 目录规划

```text
/home/ypwen/zyh/
  my_own_robodojo/        # 你的 fork 的 clone（git 仓库；独立 checkout，不与他人共用）
  # 下面这些是「可选复用」的软链接，放不放取决于导师确认（见第 4 节）
```

**纪律：git 仓库与大数据/生成物分离。** 以下都不进 git（已由 `.gitignore` 覆盖）：
`Assets/`、`.cache/`、`eval_result/`、`experiments/camera_delay/results/`、`.DS_Store`。

---

## 1. 首次部署

```bash
mkdir -p /home/ypwen/zyh
cd /home/ypwen/zyh
git clone https://github.com/PazzivalZHOU/my_own_robodojo.git
cd my_own_robodojo
git remote add upstream https://github.com/RoboDojo-Benchmark/RoboDojo.git   # 可选，便于日后同步官方
git submodule update --init --recursive
```

环境（取决于第 4 节的复用决策）：

```bash
# 若没有可复用的共享环境，才在本机自建（会产生几十 GB）
bash scripts/install.sh --install
bash scripts/init_assets.sh
bash scripts/RoboDojo/download_ckpt.sh huggingface ACT
```

> 共享账号上不要直接运行 `install.sh --install`：它可能调用 `sudo apt-get`，并在
> `$HOME/miniconda3` 中创建/修改环境，不一定局限于 `/home/ypwen/zyh/`。必须先与导师确认
> 是否使用共享环境，以及你是否拥有独立的 `$HOME`。

磁盘先看一眼：

```bash
df -h /home/ypwen/zyh
du -sh /home/ypwen/zyh/my_own_robodojo
```

---

## 2. 每次同步代码（白名单，只允许这些）

```bash
cd /home/ypwen/zyh/my_own_robodojo

git rev-parse HEAD > /home/ypwen/zyh/last_good_commit      # 记录回滚点
git fetch origin
git status                                                 # 必须干净；有改动先弄清楚来源
git merge --ff-only origin/main                            # 绝不用普通 pull 制造 merge commit

# 关键校验：hook 必须生效，否则实验会「静默跑成 baseline」
grep -n "ROBODOJO_CLIENT_ENTRY" scripts/eval_policy.sh
ls experiments/camera_delay/

# 子模块必须完整；输出行前不应有 '-' / '+' / 'U'
git submodule status --recursive
test -f third_party/IsaacLab/isaaclab.sh
```

`origin/main` 的正常同步路径应当可 fast-forward；若 `git merge --ff-only` 拒绝执行，
先停止并检查 `git status` 与 `git log --oneline --graph --decorate -10`，不要把
`reset --hard` 当作常规同步手段。没有服务器本地改动时，重新 clone 是最稳妥的恢复方式。

---

## 3. 运行实验

```bash
conda activate RoboDojo          # 或导师提供的共享环境名
cd /home/ypwen/zyh/my_own_robodojo

# 3.0 服务器 preflight：必须全部 PASS（checkpoint 名替换为实际值）
bash scripts/robodojo.sh doctor \
  --policy-dir XPolicyLab/policy/ACT \
  --task stack_blocks --ckpt <CKPT> \
  --sim-env RoboDojo --policy-env <POLICY_ENV>

# 3.1 先看命令（不运行、不写文件）
python experiments/camera_delay/run_eval.py \
  --task stack_blocks --ckpt <CKPT> --policy-env <POLICY_ENV> --dry-run

# 3.2 冒烟：只跑基线一档、1 个回合，确认能出 _result.json
python experiments/camera_delay/run_eval.py \
  --task stack_blocks --ckpt <CKPT> --policy-env <POLICY_ENV> \
  --delays 0 --eval-num 1

# 验收：driver 必须退出 0，且最新 summary 的 complete=true / eval_time>=1
python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/camera_delay/results")
summary_path = max(root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
summary = json.loads(summary_path.read_text())
run = summary["runs"][0]
assert run["complete"] is True, run
assert int(run["eval_time"]) >= 1, run
assert (summary_path.parent / "delay_0" / "_result.json").is_file()
print(f"PASS: {summary_path}")
PY

# 3.3 hook 冒烟：跑 1 帧延迟、1 个回合，确认 monkeypatch 在真实 Isaac 进程中安装
# 两条命令都必须退出 0；日志必须出现 installed: delay_frames=1
set -o pipefail
python experiments/camera_delay/run_eval.py \
  --task stack_blocks --ckpt <CKPT> --policy-env <POLICY_ENV> \
  --delays 1 --eval-num 1 2>&1 | tee /home/ypwen/zyh/camera_delay_hook_smoke.log
grep -F "[camera_delay] installed: delay_frames=1" /home/ypwen/zyh/camera_delay_hook_smoke.log

# 3.4 正式扫描（0/1/2/4；耗时约为单次评估的 4 倍）
python experiments/camera_delay/run_eval.py \
  --task stack_blocks --ckpt <CKPT> --policy-env <POLICY_ENV> \
  --delays 0,1,2,4 --eval-num <N>
```

结果：

```text
experiments/camera_delay/results/<时间戳>/
  summary.json
  delay_{0,1,2,4}/_result.json
  delay_2/policy_input/...        # 仅当 --record-policy-input
```

（更细的参数在 `configs/base.yaml` / `configs/delays.yaml`。）

### 3.5 出对比表/图

```bash
python experiments/camera_delay/analyze.py --results experiments/camera_delay/results/<时间戳>
# → analysis.md / analysis.csv / delay_vs_performance.png
# 合并多次 sweep（多 seed）：--results <stampA> <stampB>
```

---

## 4. 资源复用决策 —— 待与导师确认

先在服务器上探测是否已有共享资源：

```bash
conda env list
find / -maxdepth 6 -type d -name Eval_Layout 2>/dev/null | head
find / -maxdepth 8 -type d -path '*/policy/ACT/checkpoints' 2>/dev/null | head
```

| 资源 | 若导师有共享（推荐） | 若没有（自建） |
| :-- | :-- | :-- |
| 仿真 Assets（含 `Eval_Layout`） | 在 clone 内建软链接 `Assets -> <共享路径>`，再跑 `init_assets.sh`（会检测到已就绪而跳过下载） | 直接 `bash scripts/init_assets.sh`（会下到本仓库 `.cache/`，占你目录空间） |
| conda 环境 `RoboDojo` / 策略环境 | 直接用共享环境；不要重复 `install.sh` | `bash scripts/install.sh --install`（装在 `$HOME/miniconda3`） |
| ACT checkpoint | 软链 `XPolicyLab/policy/ACT/checkpoints -> <共享路径>`，或设 `ROBO_DOJO_POLICY_ROOT` | `bash scripts/RoboDojo/download_ckpt.sh huggingface ACT` |

> 复用前确认对共享路径有**读权限**；软链接目标由导师确认后再填。

---

## 5. 回滚

```bash
cd /home/ypwen/zyh/my_own_robodojo
git reset --hard $(cat /home/ypwen/zyh/last_good_commit)
```

- `reset --hard` **只影响被跟踪文件**，不会删 `Assets/`、`.cache/`、`eval_result/`（它们未跟踪）。
- 回滚前先 `git status` 确认没有未提交的重要改动。

---

## 6. 禁止 / 高危命令（服务器上一律不执行）

```text
git clean -fdx     # 会删除未跟踪/被忽略的 Assets、.cache、eval_result（白下几十 GB）
git clean -fd
git reset --hard   # 仅在明确回滚时、由人执行（见第 5 节）
rm -rf *           # 任何形式的批量删除
```

- 不要对子模块执行 `git checkout` / `git clean`：`XPolicyLab/policy/*/checkpoints` 是子模块内的软链接，误清会丢 checkpoint。
- 不要把服务器 SSH / 交互式终端交给 agent。

---

## 7. 一页速查

```bash
# 同步
cd /home/ypwen/zyh/my_own_robodojo && git fetch origin && git merge --ff-only origin/main

# 校验 hook
grep -n ROBODOJO_CLIENT_ENTRY scripts/eval_policy.sh

# 基线冒烟 → hook 冒烟 → 正式
python experiments/camera_delay/run_eval.py --task <T> --ckpt <C> --policy-env <E> --delays 0 --eval-num 1
python experiments/camera_delay/run_eval.py --task <T> --ckpt <C> --policy-env <E> --delays 1 --eval-num 1
python experiments/camera_delay/run_eval.py --task <T> --ckpt <C> --policy-env <E> --delays 0,1,2,4 --eval-num <N>

# 回滚
git reset --hard $(cat /home/ypwen/zyh/last_good_commit)

# 出表/出图
python experiments/camera_delay/analyze.py --results experiments/camera_delay/results/<时间戳>
```
