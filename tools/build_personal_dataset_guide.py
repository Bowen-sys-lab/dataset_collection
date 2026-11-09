"""Create a separate individual-use manual without changing collection software."""
from pathlib import Path

import build_practical_dataset_guide as layout
from docx.shared import Cm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/dataset-workflow-personal-guide.docx'
doc = layout.doc
para, heading, page = layout.para, layout.heading, layout.page
steps, table, code = layout.steps, layout.table, layout.code


def build():
    layout.setup()
    section = doc.sections[0]
    section.header.paragraphs[0].runs[0].text = '无人机双模态数据采集  个人操作手册'
    section.footer.paragraphs[0].runs[0].text = '个人版 1.1   2026年11月9日   |   '
    doc.core_properties.title = '无人机双模态数据采集个人操作手册'
    doc.core_properties.subject = '个人部署 场景接入 采集 回放 自查与保存'

    p = doc.add_paragraph(style='Title')
    layout.text_run(p.add_run('无人机双模态数据采集\n个人操作手册'), 21, True, color='000000')
    para('适合一个人在自己的电脑上完成环境搭建、场景配置、飞行采集、离线出图和数据自查。第一次按顺序做完；环境已经搭好时，直接从第5节开始。')
    para('每次采集的顺序：选好任务 → 启动场景 → 飞行并确认目标 → 按T → 关闭录制 → 离线回放 → 自查并保存。', True)
    heading('1 先知道要准备什么', 1)
    para('手册要与项目代码、完整场景资源及匹配的软件一起使用。仅有Word、几张图片或场景最外层USD，无法在新电脑完整复现。先拿到下面这些内容，再开始安装。')
    table(['需要准备', '应包含什么'], [
        ('软件与项目', '匹配的Isaac Sim安装来源；当前工程的scripts、configs及依赖配置'),
        ('场景与传感器资源', '入口USD及引用资源、无人机、贴图、MDL、红外源码、材料库、天气与光谱文件'),
        ('一个可试采的任务', '目标参考图、任务指令、起点和朝向、完成要求、场景配置与启动命令'),
        ('保存空间', '展开后的资产、原始轨迹、两路回放图片和数组；先用一条样例估算磁盘需求'),
    ], [4.1, 13.3])
    heading('当前采集参数')
    table(['项目', '个人使用时沿用的设置'], [
        ('轨迹记录', '正常飞行每1仿真秒保存一次；碰撞锁定时暂停，回退后满1秒恢复'),
        ('离线图片', '每1仿真秒选取已有位姿；RGB与红外均为256×256，宽×高'),
        ('实时预览', '沿用场景设置；RGB目标约15 FPS，固定任务入口的红外按H更新'),
        ('碰撞与终点', '人工录制默认启用碰撞回退；T另存精确终点，末段可短于1秒'),
    ], [4.1, 13.3])
    para('“仿真秒”是仿真内部经过的时间，电脑卡顿时不一定等于现实中的一秒。轨迹每秒保存一次，不会把实时预览限制成每秒一帧。')
    heading('怎样使用这份手册')
    para('配套包版本为20261109。新电脑先按《无人机双模态采集配套包安装说明》解压代码包与资源包、取得固定版本红外组件并完成检查，再阅读本手册。样例包供对照目录和输出，不计作新采集。')
    para('第2节装环境，第3至4节接场景和传感器，第5至7节完成一次真实采集，第8节自己检查，第9节整理文件。更换电脑、程序、场景或相机配置后，先重新跑通一条样例，再批量采集。')

    page('2 在自己的电脑上搭建环境')
    steps([
        '使用Windows x64和NVIDIA显卡，安装匹配的驱动。复现基线为Isaac Sim 6.1.0-rc.26+release.49347.2d230af4.gl；其他版本先验证兼容性。先确认Isaac本身能启动并使用GPU。',
        '把Isaac安装在D:\\isaacsim，把当前项目及完整资源放在D:\\IsaacSim_Demo。现有启动器仍有固定路径，首次复现优先保持这两个目录。',
        '核对红外组件和Python依赖。NumPy、SciPy、Pydantic、PyYAML、imageio、Pillow及irsim要能被Isaac自带Python导入；缺什么再补什么，不覆盖Kit、USD或Warp。',
        '关闭其他Python虚拟环境，打开PowerShell，执行下面的检查。出现FAIL时先看报告中失败项，修好后重查。',
    ])
    code(r'''Set-Location 'D:\IsaacSim_Demo'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Test-Path 'D:\isaacsim\python.bat'
Get-Content 'D:\isaacsim\VERSION'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.\handoff\dataset-workflow-kit\check_environment.ps1''')
    para('参考依赖版本：NumPy 2.3.1、SciPy 1.17.0、Pydantic 2.11.10、PyYAML 6.0.3、imageio 2.37.2、Pillow 12.3.0。以handoff/dataset-workflow-kit/environment-baseline.json为核对基线，不对已有Isaac环境一键全部升级。')
    heading('确认这些代码和目录已经拿全')
    table(['文件或目录', '用来做什么'], [
        ('scripts/record.ps1 与 process.ps1', '现场录制；离线回放、检查与建立索引'),
        ('scripts/run_isaac_with_clean_env.ps1', '使用Isaac Python并配置子进程环境'),
        ('scripts/manual_collision_undo.py\nrecording_undo.py', '碰撞停止、回退和保留路线处理，均在scripts下'),
        ('configs 与场景资产目录', '场景、任务、传感器、温度及显示配置和实际资产'),
        ('third_party/isaac-thermal-camera-simulation', '红外源码、传感器与光谱等资源'),
        ('handoff/dataset-workflow-kit', '环境检查、审核表工具、模板和工厂目标参考图'),
    ], [8.3, 9.1])
    para('软件报告保存在validation/handoff/。检查通过只说明软件和直接路径可用，还要实际开场景、试飞和回放。显卡与显存是否够用，以本机一条完整试采为准，目前没有已验证的最低硬件配置。')
    para('换盘符时要一起处理启动器、YAML、USD引用和索引路径。清理环境入口还依赖Isaac的extscache；换Isaac版本也要重验。每次新开PowerShell先点调用handoff/receiver/start_session.ps1，让City包装脚本也使用Isaac Python。旧的20261107代码ZIP已不是本次交付版本。')

    page('3 把一个场景接入采集程序')
    para('森林、工厂和城市是现有样例。后面换新场景时，主要工作是准备完整资源、填写场景YAML、核对无人机与相机路径，然后试采一条；不必先增加一个专用启动器。')
    steps([
        '收齐场景入口USD引用的subLayers、references、payload、贴图、MDL、无人机和动画。保留目录结构；远程资源须能在本机访问，或完成合规本地化并重验。',
        '在Isaac中打开入口USD，确认模型和材质齐全。核对米制、Z轴向上、无人机出生位置及净空，在Stage树复制真实Prim路径。Prim就是场景树中的节点。',
        '从工厂配置复制一份自己的YAML，按下表修改。所有相对路径均以YAML所在目录为基准；温度、天气、显示和任务文件也要对应当前场景。',
    ])
    code(r'''Set-Location 'D:\IsaacSim_Demo'
Copy-Item '.\configs\handoff_factory_example_scene.yaml' '.\configs\my_scene.yaml'
# 先编辑my_scene.yaml及它引用的配置文件 再执行下方检查''')
    table(['场景YAML字段', '填写或检查什么'], [
        ('scene.stage / thermal_scene', '场景入口USD；热环境与天气配置'),
        ('drone.prim_path / body_path', '无人机根节点；实际运动的机体节点'),
        ('drone.camera_path / minimum_height', '机体下RGB Camera；世界Z下限，不等于离地高度'),
        ('sensors.sensor_config / thermal_config', '红外传感器定义；目标温度与材料绑定'),
        ('sensors.display_config / material_directory', '红外显示方式；本场景材料库'),
        ('sensors.width / height', '实时预览尺寸；不控制离线256×256保存尺寸'),
        ('task.file', '本场景自己的任务说明YAML'),
        ('scene.collision_exclude_paths', '确需排除时填写视觉特效路径，不排除实体障碍'),
    ], [7.8, 9.6])
    code(r'''# my_scene.yaml中的路径都改好之后执行
.\handoff\dataset-workflow-kit\check_environment.ps1 -Config '.\configs\my_scene.yaml'
# 把my_scene/flight.usda替换成真实入口
& 'D:\isaacsim\python.bat' -B scripts/dataset_asset_audit.py `
  --stage 'my_scene/flight.usda' --out 'validation/handoff/my_scene-assets.json' ''')
    para('扫描报告里的unresolved和outside_project要逐项核对，区分活动缺失与非活动旧引用。通用-Config入口使用USD中的出生位姿，不会自动套用task.start_position；起点要在USD中准备，或另行适配固定起点入口。')
    para('配置复制完成后，仍需验证移动、两路相机跟随、真实障碍、回退和离线出图。静态保护器不支持直接把运动实体当静态障碍，遇到动画实体、运动父节点或动态刚体拒绝提示时，先处理适配问题。')

    page('4 配好相机和红外再开始采集')
    heading('先检查两路视点')
    steps([
        '确认RGB Camera位于实际运动的机体下。平移、转向和升降时，画面都要跟随无人机。',
        '运行时红外相机通常位于body_path/Sensors/ThermalCamera。align_thermal_to_rgb开启时使用RGB视点，仍要实际看两路图像是否对应。',
        '记录焦距、孔径、裁剪范围和相机安装位姿；修改后试采并核对每帧元数据。运行时改动不等于已保存到源USD。',
    ])
    para('当前离线RGB和红外均直接渲染为256×256，并相应调整垂直视场与有效内参。两路尺寸相同并不表示已经严格逐像素同步或完成标定；不要只用“看着差不多”代替参数记录。')
    heading('红外要固定的内容')
    table(['内容', '个人配置时怎么做'], [
        ('传感器', '沿用flir_boson_640_lwir.yaml及光谱文件；波段7.5至13.5 μm，源模型名义640×512'),
        ('天气与时刻', '固定天气CSV、地点、start_utc与环境预设，保存使用版本'),
        ('物体温度', '从已验证配置复制；使用fixed、ambient、offset模式，配置温度单位摄氏度'),
        ('物体材料', '核对目标Prim及子节点绑定、光谱和默认回退材料；RGB贴图不能代替红外材料'),
        ('显示方式', '固定温度窗、白热或黑热与色板，检查热绑定报告和实际画面'),
    ], [3.6, 13.8])
    para('按真实场景配置地面、植物、金属、玻璃和目标。不要为了让目标显眼随意增温，也不要把所有背景长期设成同一种材料。合成温度与材质设定要保存来源，不写成图片实测结果。')
    heading('分清实时预览和最终图片')
    para('工厂、森林和城市固定任务入口通常采用RGB驾驶与H红外快照；移动后停稳，再按H更新当前位置红外。参数里的“2 FPS”不表示这时在连续播放红外视频。')
    para('通用-Config入口默认连续配对预览，目标参数通常为RGB约15 FPS、红外约2 FPS，实际速度受硬件影响。预览尺寸沿用场景YAML；离线保存统一由采集默认参数控制，两者分开。')
    heading('场景碰撞也要试一次')
    para('人工录制默认使用static_sphere_undo_v2。先在短距离、已知真实障碍处验证停止与回退，然后继续飞行。森林只排除Flames和FireAtmosphere等视觉特效，桶、树干、人物和地形仍参与碰撞；不要排除实体来绕过错误。')

    page('5 准备一个任务并启动录制')
    para('第一次先用现有样例理解流程。每个任务都写清场景、任务编号、目标参考图、指令、起点与朝向、可飞范围、怎样算完成和配置版本；还没有确定的任务不要自行猜测完成规则。')
    heading('工厂样例要找什么')
    para('目标是灰色圆筒储罐，周围有蓝色护栏和红色外接管；固定白热红外中，罐壁有局部近圆形较亮区域。先用RGB核对目标身份，再用红外观察异常，在能看清目标与异常的位置停稳。弱光下同时看轮廓和周围结构。')
    image_path = ROOT / 'handoff/dataset-workflow-kit/examples/factory-target-reference.png'
    doc.add_picture(str(image_path), width=Cm(8.4))
    doc.paragraphs[-1].alignment = 1
    para('工厂参考图用于认识目标。任务YAML为configs/handoff_factory_example_task.yaml，场景配置为configs/handoff_factory_example_scene.yaml；样例不代表所有正式起点已验收。')
    heading('选择与任务对应的一种启动方式')
    code(r'''Set-Location 'D:\IsaacSim_Demo'
# 工厂教学样例 先确认本机资产 相机和USD出生位置可用
$runId = 'personal-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
.\scripts\record.ps1 -Config '.\configs\handoff_factory_example_scene.yaml' `
  -RunId $runId -HorizontalSpeed 3''')
    para('新场景用同一种-Config命令，把路径换成已经验证的my_scene.yaml。通用入口使用USD出生位姿、连续预览，升降默认0.5 m/s；工厂任务YAML中的start_position不会自动设置出生点。')
    code(r'''# 已有City C06任务的另一种示范 不与上面的命令同时运行
.\scripts\record.ps1 -Scene city -Episode C06 -CheckOnly
.\scripts\record.ps1 -Scene city -Episode C06''')
    para('Episode入口已规定起点和控制参数，不再混加自定义RunId、Config或速度。新场景名不能直接填-Scene。工厂和森林正式Episode还受就绪检查限制；遇到enabled:false或哈希不匹配，先按当前程序重新验收，不手改报告放行。')
    para('记下终端输出的完整RunId，它就是本次录制编号。重新飞一次用新RunId；同一轨迹再次回放只产生新处理批次，不算新飞行。')

    page('6 飞行时怎样操作与处理碰撞')
    heading('起飞前先检查')
    para('等待场景和传感器初始化，确认场景、出生位置和RGB画面正确。点击Drone Control取得键盘焦点；先短距离平移、升降和转向，确认相机跟随，再按任务指令搜索目标。')
    table(['按键或界面', '作用'], [
        ('W / S', '前进 / 后退'), ('A / D', '向左 / 向右移动'),
        ('R / F', '上升 / 下降'), ('Q / E', '转向'),
        ('按住X', '紧急保持；不能代替完成任务'),
        ('H', '按需模式请求当前位置红外快照；移动后重新更新'),
        ('T', '人工确认完成，另存精确终点'),
    ], [4.0, 13.4])
    heading('碰到障碍就这样处理')
    steps([
        '看到Obstacle contact: held后先松开按键。程序会停住并清除移动状态，此时T和Home均不可用。',
        '点击Undo collision。程序优先寻找碰撞前约3仿真秒的安全记录点，不足3秒时尝试最早安全点，必要时向更早安全点回退。',
        '恢复后再次点击Drone Control，重新按键，绕开障碍继续。允许多次回退，同一次录制继续使用原RunId。',
        '没有安全点、恢复失败或报错时，保留记录与日志，先排查原因；需要重飞时使用新RunId。不要关闭保护来绕过真实障碍。',
    ])
    para('回退只恢复无人机并清零速度，仿真时钟与整个世界不会倒退。原始尝试保留，离线使用修正后的保留路线；不需要手动删除碰撞或回位轨迹。')
    heading('确认目标后结束')
    steps([
        '在RGB中认准目标，停稳后更新红外；确认符合任务的完成要求，再按T一次并等保存反馈。',
        '当前回退模式下T后保持原位。正常关闭预览窗口，等待终端进程退出，不强杀、不重置场景。',
        '确认本次trajectory.jsonl和run_metadata.json已保存，再进入离线回放。漏按或误按T时注明原因，优先另开RunId重采。',
    ])
    para('停住、松开按键、按H或录制的最后一帧都不能代替T。T保存的是现场判断，回放后仍要自己复查目标和终点。')

    page('7 离线回放并检查输出')
    para('离线回放按保存的飞行位姿重新生成图片，不用再手动飞一次。先等录制进程退出；同一块GPU一次处理一条，完成后再开始下一条。')
    code(r'''Set-Location 'D:\IsaacSim_Demo'
$runId = Read-Host '粘贴本次完整RunId'
.\scripts\process.ps1 -Run $runId''')
    para('记下终端输出的实际处理目录：validation/episodes/本次RunId/处理批次/。每次重处理创建新目录，不能只记“最新一批”。正常情况下不用额外指定分辨率或取样间隔。')
    heading('先看程序有没有处理完')
    steps([
        '打开本批次pipeline_status.json，确认status为COMPLETE，回放、校验和索引阶段通过；video为手工可选项，不生成视频不代表采集失败。',
        '打开起点、转弯、目标附近和末帧的RGB及红外。核对两路来自同一批次，画面可读、相机跟随正常。',
        '确认当前两路图片都是256×256，并查看replay/replay_metadata.json的实际尺寸与采集设置。历史数据按其原有规格读取。',
        '正常观测按1仿真秒取样，精确T终点单独保留，最后一个间隔可不足1秒；不应为了凑整秒删掉终点。',
    ])
    heading('发生过回退时再多看一步')
    para('最终回放应沿保留路线继续，没有把撤销尝试、碰撞等待和回位瞬移当成连续飞行动作。对照run_metadata.json中的collection_edit_events检查记录；未解决碰撞不能导出有效序列。')
    para('回退后的观测时间是修正分支时间，source_simulation_time_s保存原始仿真时钟。原始T时间与修正STOP时间可以不同，要靠源帧和事件对应。森林动画回放按原始源时间恢复，不能用压缩后的路线时间替代。')
    heading('失败时怎样重试')
    para('保留失败目录及logs中的错误信息，先修复缺失资源、版本或运行问题，再对同一RunId重新执行process.ps1。不要把其他批次的文件补进来，也不要手改状态为PASS。Episode回放使用当次记录配置，不另加-Config覆盖。')
    table(['现象', '先检查哪里'], [
        ('缺模型或材质', '入口USD及活动引用、远程资源、本机路径'),
        ('黑图或红外不合理', '相机朝向、裁剪、温度和材料绑定、显示配置与回放日志'),
        ('终点不对应或有异常跳变', '精确T事件、源帧对应、碰撞回退记录与保留路线'),
    ], [5.1, 12.3])
    para('COMPLETE只说明程序处理完成，图像、目标和完成判断仍按第8节自查。')

    page('8 一个人怎样检查并留下结论')
    para('每条数据看完RGB和红外，并结合任务、轨迹和日志检查下面四项。自查可用、待确认、失败与重采记录分开登记；有疑问就写清问题和证据，不把不确定项写成通过。')
    table(['检查项', '要回答什么问题'], [
        ('图像可用', '有没有黑图、缺帧、错配或异常跳变；当前尺寸和视点是否正确'),
        ('目标正确', '是否真的是任务指定目标，RGB与红外是否支持判断'),
        ('路线有效', '是否符合任务；回退记录能否对应，有没有未解决碰撞或异常瞬移'),
        ('终点合理', '有没有精确T，最后的STOP是否对应任务要求的完成画面'),
    ], [3.6, 13.8])
    para('搜索途中目标不可见通常正常，两路可见性也可以不同。1秒离散轨迹和碰撞保护不能单独证明全程绝无碰撞；证据不足时如实写“无法确认”。')
    heading('个人记录只需把事实写清')
    para('可以在自己的采集记录中增加下面这些项目，或使用collection_log.csv记录已有字段、另附自查说明。个人自查说明由自己创建保存，程序不会自动生成或读取。')
    code('''采集编号 RunId：
任务编号及配置版本：
本次处理批次完整目录：
本人姓名与检查日期：
图像／目标／路线／终点：通过、发现问题或无法确认
问题证据：具体帧号、时间、文件路径及原因
个人结论：自查可用、需要重采或待确认''')
    heading('需要现有审核表时再创建')
    code(r'''Set-Location 'D:\IsaacSim_Demo'
$runId = Read-Host '输入完整RunId'
$batch = Read-Host '输入已完成处理批次的完整目录'
.\handoff\dataset-workflow-kit\create_review.ps1 -RunId $runId -Batch $batch''')
    para('工具生成review.json和frame_review.csv，新表默认为待审，不代表标注已经完成。可在整条表中记录技术、路径、终点、目标和任务检查，并保存证据；不要改写原始轨迹、任务说明或索引。')
    para('现有human_review_v1规定INCLUDE须有两名不同人员和证据。只有自己检查时，decision保持PENDING，reviewer留空，在reason中写“已完成个人自查，尚未独立复核”。个人自查可用与正式纳入是不同状态，不填虚构复核人。')
    heading('逐帧详细标注按需要做')
    para('frame_review.csv里的目标可见性、目标身份和搜索或接近阶段是可选补充项。未开展这项工作时保留默认值；发现的问题仍要记录具体帧段。检测框、分割掩码等需要另定规则。')

    page('9 把文件保存完整并方便下次找到')
    code(r'''validation/
  trajectory_runs/<RunId>/
    trajectory.jsonl           全部原始尝试
    run_metadata.json          任务 记录周期 精确T 回退事件
    training_trajectory.jsonl  有效T存在时生成的保留路线
    training_sequence_metadata.json  config_snapshot/  mileage_quality.json
  collection_launches/<RunId>/   工厂 森林启动证据
  city-v2/launches/<RunId>.json   城市启动证据
  episodes/<RunId>/<处理批次>/
    pipeline_status.json  logs/  replay/  dataset/
    derived_input/             也可能位于replay内
  annotations/<RunId>/<审核批次>/
    review.json  frame_review.csv''')
    heading('常见输出分别是什么')
    table(['文件', '含义与读取注意'], [
        ('_rgb.png / _rgb.npy', 'RGB图与数组；当前256×256，数组可能带alpha通道'),
        ('_thermal.png / _display8.npy', '红外显示图，uint8，当前256×256，可能带alpha通道'),
        ('_apparent_t.npy', '表观温度，float32，单位K；转摄氏度减273.15'),
        ('_radiance.npy / _dn16.npy', '辐亮度float32，单位看元数据；DN为uint16，不是温度'),
        ('frame_*.json / common_metadata.json', '用FrameMetadataReader合并共享字段并核对哈希'),
        ('dataset/frames.jsonl', '两路图片路径、观测时间、源帧号和事件引用'),
    ], [7.1, 10.3])
    para('世界位置单位米，Z向上，四元数顺序[w,x,y,z]。原始时间与修正时间的区别见第7节。mileage_quality.json中的corrected_branch里程是采样折线估计。dataset保存index.json、episode.json和frames.jsonl。')
    heading('每次结束后保存这些内容')
    steps([
        '保留整个原始录制目录、对应启动证据、完整成功处理批次、自查或审核记录及异常说明。不要只留PNG或视频；失败批次和待确认记录单列。',
        '记录RunId、场景与任务、本人、机器、版本、原始目录、处理批次和个人结论。collection_log.csv未包含的处理目录和自查结果，另附说明。',
        '备份这批数据引用的配置、任务参考图、场景及资源版本，保留文件清单和哈希。目录树只是示意，完整引用源也要一起保留。',
    ])
    para('跨电脑移动时，索引可能仍含旧的绝对路径。保留原索引与哈希，另建旧根目录到新目录的映射，不静默改成其他批次。正式交付时，训练、验证和测试的划分由数据负责人统一处理。')

    OUTPUT.parent.mkdir(exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    build()
