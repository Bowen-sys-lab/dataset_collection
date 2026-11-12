# City 双模态数据采集：本科生试复现版

这个仓库只提供一个 City 场景，目标是让第一次接触 Isaac Sim 的同学先跑通一条完整链路：安装环境、装入场景、检查配置、实时飞行记录轨迹、离线回放并检查 RGB/红外结果。工厂和森林场景暂不上传，也不需要为本教程准备。

这是“试复现版”，用于验证流程是否能在另一台电脑上跑起来。它不承诺不同显卡、驱动或 Isaac Sim 版本都能得到完全相同的画面；遇到问题请保留命令、版本和日志，交给项目负责人。

## 先看哪些文件

本科生按下面顺序看，其他目录先不用管：

1. `README.md`：本页安装和采集步骤。
2. `docs/dataset-workflow-personal-guide.docx`：个人采集与标注说明。
3. `handoff/receiver/start_session.ps1`：进入项目的 PowerShell 会话。
4. `tools/install_city_assets.py`：安装并校验 City 资源包。
5. `handoff/receiver/fetch_thermal.ps1`：取得固定版本红外组件并校验。
6. `tools/verify_repository.py`：检查代码、City 资源和红外组件。
7. `configs/city_200m_v3_scene.yaml`、`configs/city20_long_start_trial_v1/C06.yaml`：本次示例的场景和任务配置。
8. `scripts/record_city_v2_task.ps1`：实时预览和轨迹记录入口。
9. `scripts/process.ps1`：离线 RGB/红外回放入口。
10. `handoff/dataset-workflow-kit/examples/`、`templates/`：任务卡、记录表和交付模板。

## 电脑要求

- Windows x64、NVIDIA GPU 和可运行 Isaac Sim 的驱动；先确认 `nvidia-smi` 能正常返回。
- Git for Windows。
- 项目使用的 Isaac Sim 基线：`6.1.0-rc.26+release.49347.2d230af4.gl`，默认目录 `D:\isaacsim`。安装来源由项目负责人提供。
- 为了兼容现有启动脚本，项目目录保持为 `D:\IsaacSim_Demo`。

## 安装

在 PowerShell 中执行：

```powershell
git clone https://github.com/Bowen-sys-lab/dataset_collection.git D:\IsaacSim_Demo
Set-Location D:\IsaacSim_Demo
Set-ExecutionPolicy -Scope Process Bypass
Test-Path D:\isaacsim\python.bat
Get-Content D:\isaacsim\VERSION
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

安装仓库内的 City 资源包（约 33 MB）。它已经随 GitHub 仓库提供，不再领取工厂或森林资源：

```powershell
& D:\isaacsim\python.bat -B tools\install_city_assets.py
```

如果只想检查压缩包和已安装文件，不解压：

```powershell
& D:\isaacsim\python.bat -B tools\install_city_assets.py --check-only
```

取得固定版本红外组件并检查哈希：

```powershell
.\handoff\receiver\fetch_thermal.ps1
```

最后做一次仓库、City 资源和红外组件检查：

```powershell
& D:\isaacsim\python.bat -B tools\verify_repository.py --assets --dependency
```

看到 `"result": "PASS"` 才进入采集。红外组件来自外部仓库，首次下载需要网络并接受其使用条款。

## 先做不启动 Isaac 的检查

这一步只检查任务、场景和采集参数是否能解析：

```powershell
.\scripts\record_city_v2_task.ps1 -Task city-v2-hvac-s1 -CheckOnly
```

输出中应看到：

- 场景配置为 `configs/city_200m_v3_scene.yaml`；
- `raw trajectory=1.0 simulation seconds`，表示轨迹每 1 秒记录一次；
- `RGB=256x256 IR=256x256`，表示离线回放分辨率；
- `preview RGB=15 fps IR=2 fps`，实时预览帧率保持不变；
- `collision_undo=True`，碰撞后可以回退，原始尝试仍保留。

## 做一次真实采集

先启动实时采集：

```powershell
.\scripts\record_city_v2_task.ps1 -Task city-v2-hvac-s1
```

打开 Isaac Sim 后按窗口提示操作。用键盘控制无人机沿城市街区飞行；发生碰撞时等待回退完成，再继续。停在目标附近后按 `T` 做人工确认，然后正常关闭窗口并等待保存完成。

采集结束后记下终端打印的 `RunId`。原始轨迹位于：

```text
D:\IsaacSim_Demo\validation\trajectory_runs\<RunId>\
```

至少应有 `trajectory.jsonl`、`run_metadata.json` 和配置快照。不要删除失败尝试或碰撞前后的记录。

## 离线回放

把 `<RunId>` 换成刚才的编号：

```powershell
.\scripts\process.ps1 -Run <RunId>
```

回放会使用采集时保存的配置，按 1 秒间隔取轨迹，生成 256×256 RGB 和 256×256 红外图像，并执行文件、时间戳、位姿和任务来源检查。输出目录类似：

```text
D:\IsaacSim_Demo\validation\episodes\<RunId>\<处理时间>\
├─ replay\       # RGB、红外、元数据和 validation_summary.json
├─ dataset\      # 本次处理的 index.json
├─ logs\         # 回放和检查日志
└─ pipeline_status.json
```

看到回放元数据和 `validation_summary.json` 均为 `PASS` 后，再把该处理目录交给数据负责人。采集人员不自行划分训练集、验证集和测试集。

## 最小交付内容

每条飞行至少交付：

- `RunId`、任务编号、Git 提交号、Isaac Sim 版本、GPU/驱动版本；
- `validation/trajectory_runs/<RunId>/` 原始轨迹、启动快照和配置快照；
- `validation/episodes/<RunId>/<处理时间>/` 回放结果、`index.json`、日志和检查结果；
- 失败原因、碰撞回退次数、是否按 `T` 确认目标；
- 任务参考图或任务卡（如本次示例没有额外参考图，也要写明）。

不要把未检查的图像直接标成训练数据。`T` 只是人工确认动作，不代表自动任务成功；数据是否纳入训练由数据负责人复核决定。

## 常见问题

**找不到 USD 或贴图**：先重新运行 `tools/install_city_assets.py --check-only`，确认仓库根目录确实是 `D:\IsaacSim_Demo`，不要解压成 `D:\IsaacSim_Demo\IsaacSim_Demo`。

**红外组件缺失**：重新运行 `handoff/receiver/fetch_thermal.ps1`，确认网络可用，并检查固定提交是否与 `handoff/receiver/dependency-lock.json` 一致。

**显存不足或窗口很卡**：先只做 `-CheckOnly`；降低其他程序占用后再试。不要自行修改相机尺寸和轨迹间隔，先把日志发给负责人。

**回放失败**：保留 `logs/replay.log`、`logs/replay.stderr.log`、`replay/failure.txt` 和原始 Run 目录，不要覆盖原处理目录。

## 当前范围与待确认事项

- 本仓库只上传 City 场景；工厂、森林代码可能仍在历史目录中，但不属于本科生复现路径。
- City 源资产的完整对外分发授权仍需项目负责人确认。确认前请只在项目组范围内使用资源包。
- 尚未在所有型号的本科生电脑上完成 GPU 实测验收，因此本仓库定位是“试复现和问题收集”，不是跨硬件发布版。
- 新场景以后按同一原则增加：准备本地资产包、写场景配置、写一个 `CheckOnly` 入口、完成一次真实采集和离线回放验收后，再单独发布。

资源包版本、文件哈希和来源说明见 `resources/city/city-assets-manifest.json`、`resources/asset-packages.json` 和 `handoff/receiver/SOURCE_NOTICES.txt`。
