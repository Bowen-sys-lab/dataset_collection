"""Build the concise receiver setup companion using the existing manual design."""
from pathlib import Path
import build_practical_dataset_guide as layout

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'docs/dataset-package-setup.docx'
doc = layout.doc
para, heading, page = layout.para, layout.heading, layout.page
steps, table, code = layout.steps, layout.table, layout.code


def build():
    layout.setup()
    doc.sections[0].header.paragraphs[0].runs[0].text = '无人机双模态采集  配套包安装说明'
    doc.sections[0].footer.paragraphs[0].runs[0].text = '配套包20261109   |   '
    doc.core_properties.title = '无人机双模态采集配套包安装说明'
    p = doc.add_paragraph(style='Title')
    layout.text_run(p.add_run('无人机双模态采集\n配套包安装说明'),21,True,color='000000')
    para('供在另一台电脑部署本项目的人员使用。先用本说明装好配套包，再按《无人机双模态数据采集个人操作手册》完成一次飞行、离线出图与自查。')
    heading('1 先拿到这些文件',1)
    table(['文件', '用途与要求'],[
        ('dataset-code-20261109.zip', '必需。采集及回放代码、配置、表单、个人版与协作版手册、依赖锁定和检查工具'),
        ('dataset-assets-20261109.zip', '必需。当前工厂、森林和城市样例引用的本地USD、贴图、无人机、天气和热材质'),
        ('dataset-sample-20261109.zip', '可选。一条现有城市C20原始录制、启动记录和完整处理批次，用于对照输出'),
        ('Isaac Sim与红外组件', '单独取得。Isaac安装程序及红外上游源码不在上述ZIP内，按第2节准备'),
    ],[6.2,11.2])
    para('三个ZIP的内部顶层都叫IsaacSim_Demo，解压时合并到同一目录。先保存archives.json，它记录这次交付每个ZIP的大小和SHA256。不要把ZIP解压成多层重复的IsaacSim_Demo目录。')
    heading('本次已准备的资源')
    para('无人机cf2x及其schema已整理成本地资源，交付副本的无人机引用改为相对路径。工厂缺失的293处纹理路径从保留的原始资源补齐，逐项来源与哈希有记录。原采集工程资产没有被改写。')
    heading('先了解两个边界')
    para('本版仍要求Windows x64、NVIDIA显卡、Isaac位于D:\\isaacsim、项目位于D:\\IsaacSim_Demo。目录和版本先保持一致，跑通后再研究迁移。正式任务的就绪门禁沿用原规则，不能因拿到ZIP就视为正式启用。')
    para('样例任务用于教会部署和采集。任务设计、目标与完成规则尚需继续完善；新场景仍须按个人手册第3至4节配置并试采。')

    page('2 在新电脑按顺序安装')
    steps([
        '安装NVIDIA驱动并确认nvidia-smi可运行。安装Git for Windows，用于取得固定版本红外组件。',
        '单独安装Isaac Sim。当前基线为6.1.0-rc.26+release.49347.2d230af4.gl，安装到D:\\isaacsim。由项目负责人提供该版本可用的安装来源；若无法取得，先验证可取得版本的兼容性，不直接开展批量采集。',
        '在新目录将代码包和资源包都解压到D:\\。样例包需要时也解压到D:\\。已有工程先备份，不把本包直接覆盖到正在采集的目录。',
        '打开PowerShell，运行下方完整性检查。代码和资源的文件哈希必须通过；样例包未安装时会跳过该项。',
    ])
    code(r'''Set-Location 'D:\IsaacSim_Demo'
Set-ExecutionPolicy -Scope Process Bypass
& 'D:\isaacsim\python.bat' -B tools/verify_reproduction_delivery.py''')
    heading('取得匹配的红外组件')
    para('当前采集使用原仓库提交563a51b82938d7e03f4b41dfbbf8a6476b9e0be3。先核对原来源的使用条件，再运行下列命令。脚本从原仓库下载、检出固定提交并核对367个文件，其中9个生成的查找表由本包补充；不会改用最新版。原源码的再分发许可尚未确认，因此没有放进ZIP。')
    code(r'''.\handoff\receiver\fetch_thermal.ps1
& 'D:\isaacsim\python.bat' -B tools/verify_reproduction_delivery.py --dependency''')
    para('原仓库为https://github.com/reza-shahriari/isaac-thermal-camera-simulation。网络不可达、固定提交不可取得或哈希不一致时，先解决该项；不要用另一个irsim版本替换。脚本遇到已有不匹配目录会停止，不会删除或强制覆盖。')
    heading('检查软件环境')
    code(r'''. .\handoff\receiver\start_session.ps1
Get-Content 'D:\isaacsim\VERSION'
.\handoff\dataset-workflow-kit\check_environment.ps1 `
  -Config .\configs\handoff_factory_example_scene.yaml''')
    para('start_session.ps1前面有一个点和空格，表示只配置当前PowerShell会话。新开窗口后再运行一次。检查报告会指出缺少的Python模块；按environment-baseline.json给Isaac自带Python补齐对应依赖，不要整体升级Kit、USD或Warp。')

    page('3 先跑通一条再开始批量采集')
    para('先从工厂教学配置开始。它沿用USD里的出生位置，任务文件中的start_position不会自动改变通用入口的出生点。核对相机视点、机体高度和移动方向，再按任务参考图认目标。')
    code(r'''$runId = 'my-first-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.\scripts\record.ps1 -RunId $runId `
  -Config .\configs\handoff_factory_example_scene.yaml''')
    steps([
        '点击Drone Control再操作，按个人手册的按键表完成平移、升降和转向。先确认实时画面跟随正确，再接近任务目标。',
        '通用-Config入口默认连续配对预览；固定任务入口通常用H更新红外。正常轨迹每1仿真秒保存，实时预览帧率沿用原设置。',
        '短距离试碰撞：确认运动被锁定，点击回退按钮返回安全记录点，再继续飞。原始尝试会保留；不要把原始仿真时间当作回退后的路线时间。',
        '停稳并认准目标后按T一次，等保存反馈，再正常关闭录制。T是人工确认事件，仍要检查目标与终点。',
    ])
    heading('关闭录制后离线出图')
    code(r'''.\scripts\process.ps1 -Run $runId''')
    para('一次处理使用一个新批次。查看pipeline_status.json是否为COMPLETE；抽看起点、移动与转弯、目标附近和末帧。RGB与红外应均为256×256，定时取样约1仿真秒，精确T末帧可短于1秒。检查输出目录、数组和元数据，不只看PNG。')
    heading('通过后再采现有任务或接新场景')
    para('个人手册第5节列出现有任务入口；工厂和森林正式Episode可能仍被就绪检查拦住，按报告处理，不绕过门禁。新场景先复制配置样例，替换入口USD、无人机与相机Prim路径、热环境、温度和材料、显示及任务配置，然后重复本页试采。')
    para('接收方须自己完成一次从启动到保存的真实飞行。磁盘文件检查和本机USD检查均不能代替新电脑的GPU画面、碰撞回退和离线输出验收。')

    page('4 对照样例并保存验收记录')
    heading('随包样例是什么')
    para('样例来自现有城市C20采集，RunId为city-pilot30-C20-20261108T135934150505Z。保留原始目录与处理目录名称，既有批次仅供阅读和检查，不用它伪装成自己新采的数据，也不把它自动计为通过人工审核。')
    code(r'''validation/trajectory_runs/
  city-pilot30-C20-20261108T135934150505Z/
validation/city-v2/launches/
  city-pilot30-C20-20261108T135934150505Z.json
validation/episodes/
  city-pilot30-C20-20261108T135934150505Z/
    queue-b1601f46b0254ece9d50abf63a60931c-001/''')
    para('查看trajectory.jsonl、run_metadata.json、config_snapshot，以及处理批次内pipeline_status.json、replay和dataset。样例中的路径与元数据按原录制保留；若移动目录，保留原索引和哈希，另外记录路径映射。样例不附人工通过结论。')
    heading('留下一份能追查的验收记录')
    table(['记录项', '填写内容'],[
        ('机器与版本', 'GPU、驱动、Isaac版本、配套包版本、场景与任务配置'),
        ('实际操作', '新RunId、碰撞回退观察、T终点、原始目录、处理批次'),
        ('通过证据', '完整性报告、环境报告、COMPLETE状态、尺寸与间隔、人工抽看结果'),
        ('未通过项', '具体日志或帧、原因、修复办法和重试记录；未通过时不批量采集'),
    ],[4.3,13.1])
    para('现有表格位于handoff/dataset-workflow-kit/templates。个人自查按个人手册填写；现行正式纳入规则要求独立复核，只有自己检查时保持PENDING，不虚填复核人。')
    heading('代码与资源怎样长期共享')
    para('GitHub适合保存自有采集代码、配置模板、表单和Word手册，建议先用私有仓库。大型场景、贴图和样例数据另行存储，仓库保留版本、获取方式与SHA256清单。红外依赖保留原仓库地址和固定提交。')
    para('本次没有上传。公开之前需确认自有代码许可、工厂和城市资产来源及NVIDIA资源的分发条件；森林基础资产的CC BY署名及修改说明已保留。不要直接把整个资源ZIP作为公共仓库内容。')
    OUTPUT.parent.mkdir(exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__': build()
