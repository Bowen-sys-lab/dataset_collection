# 无人机 RGB 与红外数据采集

这个仓库用于在自己的 Windows 电脑上复现我们的双模态采集流程：**搭环境 → 接入场景 → 飞行并记录轨迹 → 离线回放生成 RGB 和红外 → 检查并保存数据**。包含当前使用的碰撞回退机制。

工厂、森林和城市是现有样例，后续场景按同一套配置方式接入。任务设计和完成规则仍在完善，样例不代表所有正式任务已经就绪。

**只克隆仓库还不能飞行：还需安装匹配的 Isaac Sim、领取场景资源包，并从原仓库取得固定版本红外组件。第一次使用按下面顺序操作。**

## 1 仓库里有什么

| 内容 | 位置 |
| --- | --- |
| 实时飞行与轨迹录制 | `scripts/record.ps1`、`scripts/start_crazyflie_dual_camera_view.py` |
| 碰撞停止与回退 | `scripts/manual_collision_undo.py`、`scripts/recording_undo.py` |
| 离线回放与检查 | `scripts/process.ps1`、`scripts/replay_trajectory_highres.py` |
| 场景、任务、相机、热环境与显示配置 | `configs/` |
| 森林专用回放及场景接入辅助代码 | `scene_intake/` |
| 采集记录、异常处理及验收表 | `handoff/dataset-workflow-kit/templates/` |
| 工厂教学任务参考图 | `handoff/dataset-workflow-kit/examples/` |
| 依赖获取、资源清单与检查工具 | `handoff/receiver/`、`tools/verify_repository.py` |
| 个人操作手册 Word | [dataset-workflow-personal-guide.docx](docs/dataset-workflow-personal-guide.docx) |
| 多人协作操作手册 Word | [dataset-workflow-practical-guide.docx](docs/dataset-workflow-practical-guide.docx) |

仓库不包含 Isaac 安装程序、第三方红外源码、场景模型与贴图包、实际录制数据。配置及辅助脚本保留了既有版本，第一次先用本文指定的教学配置，不随意挑选历史配置启动。

## 2 在新电脑安装

### 2.1 准备软件和目录

- Windows x64、NVIDIA 显卡和兼容驱动。先确认 `nvidia-smi` 和 Isaac 本身可以运行。
- 安装 Git for Windows。
- Isaac 基线：**`6.1.0-rc.26+release.49347.2d230af4.gl`**，安装目录 **`D:\isaacsim`**。向项目负责人取得该基线的安装来源；其他版本需先验证兼容性。
- 项目目录：**`D:\IsaacSim_Demo`**。目前部分启动器、天气配置与任务表仍使用这个固定路径，第一次复现保持一致。

打开 PowerShell，在尚未存在的目标目录克隆：

```powershell
git clone https://github.com/Bowen-sys-lab/dataset_collection.git D:\IsaacSim_Demo
Set-Location 'D:\IsaacSim_Demo'
Set-ExecutionPolicy -Scope Process Bypass
Test-Path 'D:\isaacsim\python.bat'
Get-Content 'D:\isaacsim\VERSION'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

已有同名采集工程时先备份，或在另一台电脑部署，不直接覆盖正在采集的文件。

### 2.2 领取并解压场景资源

向本仓库维护者或任务负责人领取 **`dataset-assets-20261109.zip`**，把它解压到 `D:\`，合并为 `D:\IsaacSim_Demo\...`。不要解压成重复两层的项目目录。

| 配套文件 | 用途 | 大小 |
| --- | --- | --- |
| `dataset-assets-20261109.zip` | 必需，工厂、森林、城市本地场景及引用资源 | 390,641,443 字节 |
| `dataset-sample-20261109.zip` | 可选，一条城市 C20 的原始录制与完整回放，供对照输出 | 62,939,117 字节 |

当前资源包**尚无仓库公开下载地址**，需要负责人另行提供。完整 SHA256 见 [asset-packages.json](resources/asset-packages.json)，逐文件清单见 [assets-manifest.json](handoff/receiver/assets-manifest.json)。没有拿到资源包时先停在环境准备阶段。

可先核对压缩包，再用资源安装检查核对解压后的每个文件：

```powershell
# 把路径替换成实际下载位置
Get-FileHash 'D:\Downloads\dataset-assets-20261109.zip' -Algorithm SHA256
```

资源包已补齐工厂缺失贴图，并将无人机 cf2x 的活动引用改为包内相对路径。来源和修改记录保存在 `handoff/receiver/texture-provenance.json` 与 `packaged-scene-check.json`。Isaac 自带的 MDL 仍由安装环境提供。

### 2.3 取得固定版本红外组件

采集代码依赖 [isaac-thermal-camera-simulation](https://github.com/reza-shahriari/isaac-thermal-camera-simulation)，固定提交：

```text
563a51b82938d7e03f4b41dfbbf8a6476b9e0be3
```

先确认原来源的使用条件，然后运行：

```powershell
.\handoff\receiver\fetch_thermal.ps1
```

脚本从原仓库下载、检出固定提交，补入本项目生成的查找表缓存，并按 [dependency-lock.json](handoff/receiver/dependency-lock.json) 校验文件。它不会覆盖已有的不匹配目录。网络或权限有问题时先解决下载，不直接换成最新版，也不要把另一个 `irsim-15015cd` 目录加入同一 `PYTHONPATH`。

### 2.4 检查安装结果

每次新开 PowerShell 先执行下面第一行。**开头的点和空格需要保留**，让当前会话中的 `python` 也使用 Isaac 启动器，城市包装脚本不会误用其他虚拟环境。

```powershell
. .\handoff\receiver\start_session.ps1

# 只检查仓库文件，不启动 Isaac
& 'D:\isaacsim\python.bat' -B tools/verify_repository.py

# 检查资源包与红外组件是否齐全
& 'D:\isaacsim\python.bat' -B tools/verify_repository.py --assets --dependency

# 检查软件模块与工厂教学配置
.\handoff\dataset-workflow-kit\check_environment.ps1 `
  -Config .\configs\handoff_factory_example_scene.yaml
```

环境报告保存到 `validation/handoff/`。缺少 Python 模块时，使用 `D:\isaacsim\python.bat` 补装缺少的应用依赖，参考版本见 [requirements-collection-reference.txt](requirements-collection-reference.txt)。例如只缺 PyYAML 时：

```powershell
& 'D:\isaacsim\python.bat' -m pip install 'PyYAML==6.0.3'
```

不要整体升级或替换 Isaac 的 Kit、USD、Warp。文件校验通过说明文件一致，环境检查通过说明对应软件和直接配置可用，仍须完成下面的实际试采。

## 3 第一次采集就按这一条跑

### 3.1 先认目标再启动

先打开 [工厂目标参考图](handoff/dataset-workflow-kit/examples/factory-target-reference.png)。教学样例寻找灰色圆筒储罐，周围有蓝色护栏和红色外接管，红外中罐壁有局部较亮区域。

```powershell
Set-Location 'D:\IsaacSim_Demo'
. .\handoff\receiver\start_session.ps1
$runId = 'my-first-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
.\scripts\record.ps1 -RunId $runId `
  -Config .\configs\handoff_factory_example_scene.yaml `
  -HorizontalSpeed 3
```

通用 `-Config` 入口使用 USD 中保存的出生位置，**不会自动套用任务 YAML 的 `start_position`**。等待场景加载，确认出生点和相机画面正常，再点击 **Drone Control** 获得键盘焦点。

| 按键或界面 | 用途 |
| --- | --- |
| W / S | 前进 / 后退 |
| A / D | 向左 / 向右移动 |
| R / F | 上升 / 下降 |
| Q / E | 转向 |
| 按住 X | 紧急保持 |
| H | 按需红外模式下，请求当前位置红外快照 |
| T | 人工确认完成并保存精确终点 |
| Undo collision | 碰撞锁定后退回安全记录点 |

先短距离平移、升降和转向，确认两路视点跟随机体。通用 `-Config` 默认连续配对预览；现有固定任务入口通常为 RGB 驾驶、按 H 更新红外。按需模式中，移动后先停稳，再更新红外。

### 3.2 碰撞后怎样继续

看到 `Obstacle contact: held` 后松开运动键。点击 **Undo collision**，程序优先返回碰撞前约 3 仿真秒的安全记录点，然后重新点击 Drone Control，绕开障碍继续。可以在同一个 RunId 内多次回退。

碰撞锁定时 T 和 Home 不可用。回退只恢复无人机状态，世界和原始仿真时钟不会倒转；原始尝试保留，离线处理使用修正后的保留路线。没有安全点或恢复失败时保留日志排查，重飞使用新 RunId。

### 3.3 确认目标并结束

在 RGB 中认准目标，停稳后查看红外，符合任务完成要求时按 **T 一次**。等保存反馈后正常关闭录制窗口，并等待进程退出。记下完整 RunId，不强杀进程。

T 保存人工确认事件，并不自动证明找对目标或完成任务。回放后仍需检查目标、路线和终点。

## 4 离线回放出图

等录制退出后运行，同一块 GPU 一次处理一条：

```powershell
.\scripts\process.ps1 -Run $runId
```

每次重处理建立新批次，实际目录以终端输出为准，通常为 `validation/episodes/<RunId>/<处理批次>/`。

检查 `pipeline_status.json` 的状态为 **COMPLETE**，再抽看起点、移动和转弯、目标附近以及末帧。两路画面、数组和元数据都要保留，不能只交 PNG 或视频。视频需要时单独生成，不是离线采集的必选步骤。

| 当前设置 | 数值与含义 |
| --- | --- |
| 原始轨迹记录 | 正常飞行每 **1 仿真秒**一次，碰撞锁定时暂停 |
| 离线取样 | 每 **1 仿真秒**选取已有位姿，精确 T 末帧允许不足 1 秒 |
| 离线 RGB | **256 × 256**，宽 × 高 |
| 离线红外 | **256 × 256**，宽 × 高 |
| 实时预览 | 保持场景设置，RGB 目标约 15 FPS；固定任务通常用 H 更新红外 |

仿真秒不一定等于现实秒。1 Hz 轨迹保存不会把实时预览降到每秒一帧；按需红外参数中的 2 FPS 也不代表连续红外视频。

回退后的观测时间属于修正分支，`source_simulation_time_s` 保留原始时钟。森林动画回放需要这个源时间，不要手工删改轨迹或重新编号来处理回退。

## 5 数据保存在哪里

```text
validation/
  trajectory_runs/<RunId>/
    trajectory.jsonl                  原始轨迹和尝试
    run_metadata.json                 任务 记录周期 T终点 回退事件
    config_snapshot/                  录制时配置
    training_trajectory.jsonl          有效T存在时生成的保留路线
    training_sequence_metadata.json
    mileage_quality.json
  collection_launches/<RunId>/         工厂和森林的启动证据
  city-v2/launches/<RunId>.json         城市启动证据
  episodes/<RunId>/<处理批次>/
    pipeline_status.json
    logs/  replay/  dataset/
    derived_input/                    也可能位于replay内
  annotations/<RunId>/<审核批次>/
    review.json  frame_review.csv
```

回放包含 RGB 图片及数组、红外显示图、表观温度、辐亮度、DN 和元数据。`_apparent_t.npy` 单位为 K，转摄氏度减 273.15。世界位置单位米、Z 向上，四元数顺序为 `[w,x,y,z]`。详细字段见个人 Word 手册第 9 节。

完成后保存原始录制、启动证据、选用的完整成功处理批次、检查记录和异常说明。失败和待确认记录单列。个人自查记录目标、路线、图像与终点；使用现有审核表时，独自检查保留 `PENDING`，不虚填独立复核人。数据集划分由数据负责人统一处理。

## 6 接入自己的新场景

不必先为每个新场景写启动程序。先复制 [工厂教学配置](configs/handoff_factory_example_scene.yaml)，修改为自己的配置：

```powershell
Copy-Item .\configs\handoff_factory_example_scene.yaml .\configs\my_scene.yaml
```

| 配置项 | 要准备的内容 |
| --- | --- |
| `scene.stage` | 入口 USD 及完整的 subLayers、references、payload、贴图和 MDL 引用 |
| `scene.thermal_scene` | 热环境、天气文件、时段与位置 |
| `drone.prim_path` / `body_path` | 无人机根节点和实际运动机体节点 |
| `drone.camera_path` | 位于运动机体下面的 RGB Camera 路径 |
| `sensors.sensor_config` | 红外传感器及光谱文件 |
| `sensors.thermal_config` / `material_directory` | 本场景温度、热材质及目标绑定 |
| `sensors.display_config` | 红外显示范围与色板 |
| `sensors.width` / `height` | 实时预览尺寸，离线尺寸由采集默认参数单独控制 |
| `task.file` | 自己的任务指令、目标说明和完成要求 |
| `scene.collision_exclude_paths` | 仅需排除的视觉特效，不排除实际障碍 |

`Prim` 是 USD 场景树中的节点。在 Isaac 中复制真实节点路径，不根据名字猜测。相对路径通常以对应 YAML 所在目录解析；现有天气配置含固定绝对路径，换场景也要一起核对。

改完后检查：

```powershell
.\handoff\dataset-workflow-kit\check_environment.ps1 -Config .\configs\my_scene.yaml

# 将stage替换成实际入口，输出报告使用新文件名
& 'D:\isaacsim\python.bat' -B scripts/dataset_asset_audit.py `
  --stage 'my_scene/flight.usda' --out 'validation/handoff/my-scene-assets.json'
```

资产扫描包含非活动引用和 Isaac 内置 MDL，出现 `unresolved` 要逐项核对，不能只看数量判断成功。接着按第 3 至 4 节试采，检查移动、两路相机跟随、碰撞回退、T 终点和离线图像。当前碰撞保护面向静态障碍，动态实体或运动父节点等适配问题需另行处理。

## 7 现有固定任务与常见问题

城市任务示例：

```powershell
.\scripts\record.ps1 -Scene city -Episode C06 -CheckOnly
.\scripts\record.ps1 -Scene city -Episode C06
```

固定任务入口自带起点、控制参数和指令，不与自定义 RunId、Config 或速度混用。工厂与森林正式 Episode 还受原有就绪证据和哈希门禁限制，仓库没有携带完整正式验收证据。遇到禁用或哈希不匹配时由负责人重新验证，不直接修改成通过。

| 问题 | 先检查 |
| --- | --- |
| 缺 USD、贴图或相机 | 资源包是否拿全，是否解压错层级，Prim 路径是否真实存在 |
| 无法导入 irsim | 固定版本是否下载完成，是否混用新旧版本或其他 Python 环境 |
| 软件路径不对 | Isaac 是否在 `D:\isaacsim`，项目是否在 `D:\IsaacSim_Demo` |
| 城市启动误用 Python | 新会话是否点调用 `start_session.ps1` |
| 红外不合理或图像黑屏 | 温度、热材质、显示范围、相机朝向与日志 |
| 碰撞后不能按 T | 尚处于锁定状态，先回退并恢复控制 |
| 离线失败 | 保留失败目录和 logs，修复后对同一 RunId 建新处理批次 |

## 8 开发检查和版本记录

不依赖场景资产的基本回归：

```powershell
& 'D:\isaacsim\python.bat' -B -m unittest `
  tests.test_recording_undo tests.test_manual_collision_undo `
  tests.test_episode_replay_source tests.test_dataset_handoff `
  tests.test_reproduction_delivery tests.test_repository_setup
```

场景包和依赖安装后，再运行包含实际配置的检查：

```powershell
& 'D:\isaacsim\python.bat' -B -m unittest tests.test_collection_capture_revision
```

上述测试不启动完整 GPU 采集，不能替代新电脑的真实飞行验收。批量工作前，在自己的电脑完成一条从启动、飞行、回退、T 到离线出图和自查的记录。

复现时记录 Git 提交号、资源包版本、Isaac 和驱动版本、场景与任务版本、RunId 和处理批次。Word 手册按 20261109 本地交付包编制，**仓库部署与资源获取以本 README 为准**。`tools/build_reproduction_delivery.py` 等是原本地交付工具，重新生成全量交付包需要原始工程和资产，不是克隆后的安装命令。

维护者有意修改仓库文件后，运行 `D:\isaacsim\python.bat -B tools/seal_repository_manifest.py` 更新仓库文件清单，再重新检查和提交。接收方遇到意外哈希变化时先核对 Git 状态，不靠重新生成清单掩盖问题。

## 9 资源来源与使用条件

红外源码从其原仓库取得，固定提交未发现 LICENSE，使用与分发条件须向原来源确认；本仓库没有给第三方软件授予新许可。森林基础资产的署名和 CC BY 4.0 说明保留在资源包中；NVIDIA 无人机和人物遵循各自资源条款。工厂、城市资产的完整对外分发授权尚未在本次整理中核实，因此资源包未上传到本仓库。

具体来源见 [SOURCE_NOTICES.txt](handoff/receiver/SOURCE_NOTICES.txt)。仓库公开可访问不表示所有外部软件、场景和数据都可以任意再分发。
