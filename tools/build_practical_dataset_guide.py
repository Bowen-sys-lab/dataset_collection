"""Build the role-based, shareable collection manual from current project facts."""
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/dataset-workflow-practical-guide.docx'
doc = Document()


def text_run(run, size=10.5, bold=False, font='Microsoft YaHei', color='222222'):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    fonts = run._element.get_or_add_rPr().rFonts
    fonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    return run


def para(text, bold=False):
    p = doc.add_paragraph()
    text_run(p.add_run(text), bold=bold)
    return p


def heading(text, level=2):
    p = doc.add_paragraph(style=f'Heading {level}')
    text_run(p.add_run(text), 15 if level == 1 else 11.5, True, color='000000')
    return p


def page(title):
    p = heading(title, 1)
    p.paragraph_format.page_break_before = True


def bullets(items):
    for text in items:
        p = doc.add_paragraph(style='List Bullet')
        text_run(p.add_run(text))


def steps(items):
    # A fresh native Word numbering instance makes each procedure start at 1.
    numbering = doc.part.numbering_part.element
    ids = [int(n.get(qn('w:abstractNumId'))) for n in numbering.findall(qn('w:abstractNum'))]
    abstract_id = max(ids, default=-1) + 1
    abstract = OxmlElement('w:abstractNum')
    abstract.set(qn('w:abstractNumId'), str(abstract_id))
    kind = OxmlElement('w:multiLevelType'); kind.set(qn('w:val'), 'singleLevel'); abstract.append(kind)
    level = OxmlElement('w:lvl'); level.set(qn('w:ilvl'), '0')
    for tag, value in [('start', '1'), ('numFmt', 'decimal'), ('lvlText', '%1.'), ('lvlJc', 'left')]:
        node = OxmlElement('w:' + tag); node.set(qn('w:val'), value); level.append(node)
    props = OxmlElement('w:pPr')
    indent = OxmlElement('w:ind'); indent.set(qn('w:left'), '330'); indent.set(qn('w:hanging'), '330')
    props.append(indent); level.append(props); abstract.append(level)
    first_num = numbering.find(qn('w:num'))
    if first_num is None: numbering.append(abstract)
    else: numbering.insert(numbering.index(first_num), abstract)
    num_id = max([int(n.get(qn('w:numId'))) for n in numbering.findall(qn('w:num'))], default=0) + 1
    num = OxmlElement('w:num'); num.set(qn('w:numId'), str(num_id))
    link = OxmlElement('w:abstractNumId'); link.set(qn('w:val'), str(abstract_id)); num.append(link); numbering.append(num)
    for text in items:
        p = doc.add_paragraph(style='List Paragraph')
        pr = p._p.get_or_add_pPr(); npr = OxmlElement('w:numPr')
        ilvl = OxmlElement('w:ilvl'); ilvl.set(qn('w:val'), '0')
        nid = OxmlElement('w:numId'); nid.set(qn('w:val'), str(num_id))
        npr.extend([ilvl, nid]); pr.append(npr)
        text_run(p.add_run(text))


def code(text):
    p = doc.add_paragraph(style='Code')
    text_run(p.add_run(text.strip()), size=8.2, font='Consolas')
    p.paragraph_format.keep_together = True


def table(headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Table Grid'; t.autofit = False
    for col, width in zip(t.columns, widths): col.width = Cm(width)
    pr = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        b = OxmlElement('w:' + edge)
        for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]: b.set(qn('w:' + key), value)
        borders.append(b)
    pr.append(borders)
    for i, values in enumerate([headers] + list(rows)):
        cells = t.rows[0].cells if i == 0 else t.add_row().cells
        trpr = cells[0]._tc.getparent().get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
        if i == 0: trpr.append(OxmlElement('w:tblHeader'))
        for cell, value, width in zip(cells, values, widths):
            cell.width = Cm(width); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp = cell._tc.get_or_add_tcPr(); margin = OxmlElement('w:tcMar')
            for tag, val in [('top', '65'), ('bottom', '65'), ('left', '85'), ('right', '85')]:
                n = OxmlElement('w:' + tag); n.set(qn('w:w'), val); n.set(qn('w:type'), 'dxa'); margin.append(n)
            cp.append(margin)
            if i == 0:
                shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), 'EAEFF3'); cp.append(shd)
            cell.text = ''; p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = 1.06
            text_run(p.add_run(str(value)), 9, i == 0)
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1; text_run(p.add_run(''), 2)


def setup():
    sec = doc.sections[0]; sec.page_width = Cm(21); sec.page_height = Cm(29.7)
    sec.top_margin = Cm(1.7); sec.bottom_margin = Cm(1.65)
    sec.left_margin = Cm(1.8); sec.right_margin = Cm(1.8)
    sec.header_distance = Cm(.65); sec.footer_distance = Cm(.65)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'List Paragraph', 'List Bullet']:
        s = doc.styles[name]; s.font.name = 'Microsoft YaHei'; s.font.size = Pt(10.5)
        s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
        s.font.color.rgb = RGBColor.from_string('000000' if name in ['Title', 'Heading 1', 'Heading 2'] else '222222')
        s.paragraph_format.space_after = Pt(4); s.paragraph_format.line_spacing = 1.12
        if name.startswith('Heading') or name == 'Title':
            s.paragraph_format.keep_with_next = True
            s.paragraph_format.space_before = Pt(5); s.paragraph_format.space_after = Pt(6)
        pp = s._element.get_or_add_pPr()
        for b in list(pp.findall(qn('w:pBdr'))): pp.remove(b)
    cs = doc.styles.add_style('Code', 1); cs.font.name = 'Consolas'; cs.font.size = Pt(8.2)
    cs.paragraph_format.line_spacing = 1.0; cs.paragraph_format.space_after = Pt(7)
    cs.paragraph_format.space_before = Pt(3); cs.paragraph_format.left_indent = Cm(.15)
    text_run(sec.header.paragraphs[0].add_run('无人机双模态数据集  采集操作与环境复现手册'), 8, color='000000')
    f = sec.footer.paragraphs[0]; f.alignment = 2
    text_run(f.add_run('版本 2.1   2026年11月9日   |   '), 8, color='666666')
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); f._p.append(field)
    doc.core_properties.title = '无人机双模态数据集采集操作与环境复现手册'
    doc.core_properties.subject = '采集人员操作 标注复核 新电脑部署 新场景接入'
    doc.core_properties.author = '项目数据采集团队'


def build():
    setup()
    p = doc.add_paragraph(style='Title')
    text_run(p.add_run('无人机双模态数据集\n采集操作与环境复现手册'), 21, True, color='000000')
    para('采集人员从本页开始，照第1至4节完成一次采集；标注和复核人员看第5至6节。环境部署与新增场景由工程人员按附录A至F处理。')
    para('日常只记这条顺序：领取任务 → 打开场景 → 飞行并确认目标 → 按T保存终点 → 关闭并回放 → 检查和提交。', True)
    heading('1 先分清哪些事情只做一次', 1)
    table(['什么时候做', '谁来做', '完成后交给采集员什么'], [
        ('每台电脑首次部署', '工程人员', '装好软件和资源，完成一次真实试采；见附录A'),
        ('每个场景首次接入或更新', '工程与任务人员', '可用场景、相机和红外配置、任务卡、启动命令；见附录B至D'),
        ('每次实际采集', '采集人员', '按第2至4节操作，保留RunId、原始记录与处理批次'),
        ('每条数据审核与交付', '标注员和复核人', '按第5至6节看图、填表、复核；保留真实结论'),
    ], [4.3, 3.0, 10.1])
    para('日常采集不必重新安装软件或修改场景配置。电脑、程序、场景或传感器版本变更时，由工程人员重新验收后再开工。')
    heading('当前固定参数')
    table(['项目', '当前约定'], [
        ('轨迹记录', '正常飞行每1仿真秒保存一次；碰撞锁定时暂停，回退后满1秒恢复'),
        ('离线回放', '每1仿真秒选已有位姿；RGB和红外均默认256×256，宽×高'),
        ('实时预览', '沿用场景尺寸与帧率。RGB目标约15 FPS；任务入口的红外按H更新'),
        ('碰撞回退', '人工录制默认开启；支持当前工厂、森林、城市的共用入口'),
        ('完成终点', '现场按T另存精确时刻，最后一个图像间隔可以短于1秒'),
    ], [3.4, 14.0])
    para('RunId就是这一次实际录制的编号。重飞会产生新RunId；同一条轨迹重新回放只是新处理批次，不算新飞行。')

    page('2 领取任务并开始飞行')
    heading('开工前先拿到任务卡')
    para('负责人应提供场景和任务编号、目标参考图、指定起点、任务指令、怎样算完成、启动命令，以及交付位置。缺少这些信息时先补齐，不自行猜目标或修改参数。')
    heading('在项目目录启动')
    code(r'''Set-Location 'D:\IsaacSim_Demo'
# 本窗口若禁止运行脚本 可执行一次 关闭窗口后失效
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# 以City C06示范 换成任务卡给出的场景与编号
.\scripts\record.ps1 -Scene city -Episode C06 -CheckOnly
.\scripts\record.ps1 -Scene city -Episode C06''')
    para('先读预检结果，再启动第二条命令。City预检核对任务、起点和配置；工厂与森林还受正式就绪检查限制。若出现enabled:false、哈希不匹配或错误，交工程人员处理，不跳过检查。')
    para('当前人工录制的碰撞回退默认开启，不必额外加-CollisionUndo。新场景由工程人员给出已验证的-Config命令；不要把自己的场景名直接填到-Scene。')
    steps([
        '记下终端的完整RunId，在collection_log.csv登记任务编号和采集人。',
        '等待场景初始化，检查目标场景、出生位置、RGB画面及任务是否正确；点击Drone Control取得键盘焦点。',
        '先做短距离移动、升降和转向，确认相机跟随、画面正常；再沿任务要求搜索和接近目标。',
    ])
    table(['按键或界面', '作用'], [
        ('W / S', '前进 / 后退'), ('A / D', '向左 / 向右移动'),
        ('R / F', '上升 / 下降'), ('Q / E', '转向'),
        ('按住X', '紧急保持，不能当作完成任务'),
        ('H', '按需模式请求当前位置红外快照；移动后重新按H'),
        ('T', '人工确认完成，另存精确终点；具体见第3节'),
    ], [4.0, 13.4])
    para('红外旧画面不代表无人机当前看到的内容。正式任务入口采用按需快照时，停稳后按H再判断；任务入口的“2 FPS”参数不表示持续播放的红外视频。')

    page('3 碰撞时回退并在目标处结束')
    heading('碰撞后按这个顺序处理')
    steps([
        '碰到障碍时，程序自动停住并清除移动按键。看到Obstacle contact: held后先松开按键；此时不能按T，Home重置也被禁用。',
        '点击Undo collision按钮。程序寻找碰撞前约3仿真秒的安全记录点；起飞不足3秒时尝试最早安全点，必要时寻找更早的安全点。',
        '回退完成后点击Drone Control，重新按键，绕开障碍继续飞。可以重复回退，继续使用同一个RunId。',
        '没有安全点、恢复失败或程序报错时，保留当前记录与日志，通知工程人员；需要重飞就使用新RunId。',
    ])
    para('回退只恢复无人机位姿并清零速度，仿真时钟不倒退。原始尝试始终保留；离线处理自动使用保留的有效路线，不需要采集员删轨迹。')
    heading('找到目标后再按T')
    steps([
        '先在RGB中核对目标身份，再停稳按H核对当前红外。以任务卡的完成规则为准，不凭一个亮点判断。',
        '确认完成后按T一次，等终端或界面的保存反馈。当前回退模式下T后保持原位；不要继续把额外移动当成示范。',
        '正常关闭预览窗口，等待终端进程退出。不要强杀进程、重置场景或手动倒放时间轴。',
        '检查validation/trajectory_runs/本次RunId/下的trajectory.jsonl和run_metadata.json已保存，再登记完成情况。',
    ])
    heading('遇到问题先这样做')
    table(['现象', '处理'], [
        ('按键没反应', '点击Drone Control，确认窗口焦点'),
        ('移动后红外没变化', '按需模式停稳后重新按H；仍异常则保留日志'),
        ('漏按T或误按T', '保留本次尝试并注明原因，通常另开RunId重采'),
        ('缺模型、黑图、穿模或明显卡顿', '记录RunId和发生时间，停止批量采集并交工程人员排查'),
    ], [5.0, 12.4])
    para('按H、停住、松开按键或最后一条轨迹都不能代替T完成事件。现场T记录了操作员判断，数据后续仍需人工复核。')

    page('4 离线出图并提交本次数据')
    para('“离线回放”是按保存的飞行位姿重新生成图片，不用手动再飞一遍。等录制进程退出后执行；同一块GPU逐条回放，上一条完成后再开始下一条。')
    code(r'''Set-Location 'D:\IsaacSim_Demo'
$runId = Read-Host '粘贴终端给出的完整RunId'
.\scripts\process.ps1 -Run $runId''')
    para('终端会输出本次处理目录：validation/episodes/本次RunId/处理批次/。重处理会创建新目录；把这个实际目录一并记入采集日志，不能只写“最新那一批”。')
    heading('提交前检查这五项')
    steps([
        '本批次pipeline_status.json的status为COMPLETE，回放、校验和索引阶段均通过。失败批次保留，修复后另建目录重试。',
        '打开起点、转弯、目标附近和末帧的RGB/红外，确认画面可读、视点合理、配对没拿错。',
        '本批次RGB与红外图片均为256×256；实际尺寸也在replay/replay_metadata.json中。旧数据按原批次尺寸读取。',
        '终点与现场T一致；有回退时使用修正分支，并保留原始记录和回退事件。没有T则注明未确认，不能冒充已完成。',
        '采集日志写齐RunId、场景/任务、采集人、处理目录、是否按T、异常及交付位置。',
    ])
    heading('交出这些文件')
    table(['交付内容', '保存位置'], [
        ('完整原始录制', 'validation/trajectory_runs/本次RunId/'),
        ('当次启动与配置来源', '工厂/森林：collection_launches/；城市：city-v2/launches/，均在validation下'),
        ('完整成功处理批次', 'validation/episodes/本次RunId/处理批次/，含replay、dataset、logs和状态文件'),
        ('采集日志与问题', 'collection_log.csv及引用的截图、时间和异常说明'),
    ], [5.0, 12.4])
    para('以上目录要连同其引用的场景与配置版本交付，不能只发PNG或视频。标注人员接手后按第5至6节审核；COMPLETE只说明处理完成，不等于任务或训练数据已经验收。')
    heading('失败后怎么重试')
    para('先保留错误日志和失败目录。由工程人员修复资源、版本或运行问题，再对同一RunId重新执行process.ps1。不要把其他运行的文件补进来，也不要手改报告为PASS。')

    page('5 审核人员每条数据必须检查什么')
    para('每条数据先做四项检查：图像可用、目标正确、路线有效、终点合理。逐帧可见性和飞行阶段是按需补充的标注，不要求采集人员默认逐帧填写；检测框和分割掩码需另定规则。')
    heading('先创建本条数据的审核表')
    code(r'''Set-Location 'D:\IsaacSim_Demo'
$runId = Read-Host '输入要审核的完整RunId'
$batch = Read-Host '输入已完成处理批次的完整目录'
.\handoff\dataset-workflow-kit\create_review.ps1 `
  -RunId $runId -Batch $batch''')
    para('现有工具能生成review.json整条审核表和frame_review.csv逐帧表，但不会自动判断图像或任务是否成功。新表中的PENDING、UNKNOWN、NOT_REVIEWED都表示尚未确认，不能当成已完成标注。表单绑定来源哈希；不要手改原始轨迹、录制时保存的任务说明或自动索引。')
    steps([
        '查看本次录制保存的任务说明、目标参考图和完成要求，确认场景、任务编号、要找的目标及版本。',
        '按索引顺序看完RGB和红外，检查黑图、画面缺失、错配和异常跳变。发现问题时记录具体帧号、时间或文件路径，无需默认逐帧填可见性和阶段。',
        '结合采集日志和轨迹检查保留路线是否符合任务，有无未解决的碰撞或回退异常。搜索途中目标不可见通常正常，两路可见性也可以不同。',
        '找到精确T终点和对应STOP，确认画面支持任务卡的完成判断。缺T、错目标或判断不清都如实记录。',
        '填写review.json中的技术、路径、终点、目标和任务结论；交给第二人复核。分歧未解决时保持PENDING。',
    ])
    heading('碰撞回退的数据多看一步')
    para('对照collection_edit_events确认碰撞和回退已有对应记录、没有未解决碰撞，最终回放只含保留路线。不要把原始回退瞬移作为飞行动作，也不要因原始尝试中有被撤销段就直接删掉整次飞行。')
    para('有回退时同时核对原始T时间与修正STOP时间，用source_simulation_time_s和事件引用对应；两种时间不相等不一定是错误。最终路线里程看mileage_quality.json的corrected_branch。')
    heading('填完后检查表单')
    code(r'''$review = Read-Host '输入生成的review.json完整路径'
& 'D:\isaacsim\python.bat' -B -m scripts.dataset_handoff `
  verify-review --review $review''')
    para('verify-review核对整条审核表的声明值和来源完整性，不会检查图片内容，也不会验证逐帧CSV是否填对。按需编辑CSV时保留UTF-8、原帧号、时间与路径；JSON保持合法格式。')

    page('6 填写审核结论并完成交付')
    heading('每条数据必做的审核记录')
    table(['review.json字段', '填写内容'], [
        ('technical_check / path_check\nterminal_check', 'PASS、FAIL或UNKNOWN，分别表示技术、路径、终点检查'),
        ('target_correct / task_success', 'YES、NO或UNKNOWN；依据任务卡，不由T自动填YES'),
        ('terminal_source / terminal_time_s\nterminal_original_frame_id', '真实事件来源、明确说明时间轴的终点时间、可追溯原帧号；回退前后时间可补写在evidence和reason'),
        ('decision / reason', 'INCLUDE纳入、EXCLUDE排除、PENDING待审及理由'),
        ('annotator / reviewer / reviewed_at', '真实标注人、独立复核人和日期；两人不能同名'),
    ], [6.0, 11.4])
    para('技术、路径和终点均PASS，正确目标及任务完成均YES，问题已解决，有证据且两人复核一致，才填INCLUDE。明确错误填EXCLUDE并写原因；信息不足或意见不一填PENDING。路径证据不足时保留UNKNOWN，不把碰撞保护或1秒离散轨迹当成全程无碰撞证明。')
    para('STOP表示明确完成事件，HOLD表示停留，两者不相同。漏按T优先重采；飞后补标必须另存来源与依据，不能伪装成现场按T。')
    heading('逐帧标注按研究需要补充')
    para('frame_review.csv是备用表。rgb_target_visibility与thermal_target_visibility记录两路目标可见性，identity_confirmed记录目标身份，phase记录搜索、接近等阶段。任务负责人要求时先下发判定规则，再逐帧填写；未开展时保留默认值，不影响完成整条审核，也不表示该帧不能用于训练。')
    para('发现的问题可记在CSV的issue_code、evidence、annotator_note或整条审核表中，写清具体帧段、文件路径与原因。')
    heading('提交要求')
    para('提交时保留完整原始记录、回放结果、审核表和异常说明，并写清采集编号与处理目录。未通过审核的数据单独存放。训练集、验证集和测试集由数据负责人统一划分，采集人员不用自行分配。')

    page('附录A 每台电脑首次搭建环境')
    para('由工程人员执行。交付方先准备匹配的Isaac安装来源、当前工程代码、完整场景资产和依赖，再在接收方自己的电脑部署。工程不是仅靠一个脚本或最外层USD就能复现。')
    code(r'''D:\isaacsim\             Isaac Sim安装目录
D:\IsaacSim_Demo\        项目目录
  scripts\  configs\  scene_intake\
  third_party\isaac-thermal-camera-simulation\
  handoff\dataset-workflow-kit\
  factory\  forest\ 等实际场景资产
  validation\            录制 回放 标注和检查报告''')
    steps([
        '安装Windows x64与适配的NVIDIA驱动。复现基线为Isaac Sim 6.1.0-rc.26+release.49347.2d230af4.gl；先确认Isaac自身可启动并能使用GPU。其他版本先做兼容验证。',
        '将当前项目、场景、无人机、纹理、热环境、红外材质与光谱数据放到约定目录。各项核对文件清单和版本，不复用原电脑的个人缓存。',
        '依赖安装到Isaac自带Python。核对NumPy、SciPy、Pydantic、PyYAML、imageio、Pillow和irsim；参考environment-baseline.json。不要另装USD或Warp覆盖Isaac组件。',
        '关闭其他虚拟环境，在项目目录执行下列检查。缺包由工程人员按交付基线补齐，检查通过后继续真实试采。',
    ])
    code(r'''Set-Location 'D:\IsaacSim_Demo'
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Get-Content 'D:\isaacsim\VERSION'
Test-Path 'D:\isaacsim\python.bat'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.\handoff\dataset-workflow-kit\check_environment.ps1''')
    para('软件检查报告位于validation/handoff/，它不启动Kit或证明完整资源可加载。显卡型号、显存、驱动、工程版本和运行耗时记入部署表；尚无经验证的最低硬件配置，以本机完整试采结果为准。')
    heading('目录和依赖怎么处理')
    para('现有启动器固定D盘目录，并使用Isaac extscache中的omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312。换盘符或Isaac版本时，工程人员要一起修改启动器、YAML、USD引用及历史索引映射并重验；只切换终端目录不够。')
    para('run_isaac_with_clean_env.ps1给子进程设置USD DLL路径、PYTHONPATH和Warp缓存。不要把其他Python或USD目录堆进全局PATH。磁盘按展开资产、一条完整回放的实测占用、计划条数和重处理余量估算。')
    para('City的PowerShell包装脚本另调用PATH中的python解析YAML。工程人员需确认该命令可用且装有PyYAML；仿真本身仍使用Isaac Python。')
    para('第三方组件和场景由交付方确认可用来源及交付权限，保留许可证。旧代码ZIP不是新版发布包；交付前须重新整理版本和清单，见附录F。')

    page('附录B 每个新场景怎样接入')
    para('由工程人员执行。最省事的方法是复制一个已跑通的配置，逐项换成新场景的真实资源，然后先录一条短样例。新增场景先用-Config入口，不必立即做专用启动器。')
    steps([
        '从入口USD递归收齐subLayers、references、payload、无人机、贴图、MDL与动画。保持目录结构；远程引用要确认能访问，或正规本地化后重新验证。',
        '复制configs/handoff_factory_example_scene.yaml为新场景YAML，并准备该场景自己的任务、热环境、热配置和显示配置。所有相对路径以YAML所在目录为基准。',
        '在Isaac Stage中核对下表路径和米制、Z轴向上。不要按显示名称猜Prim路径。',
    ])
    table(['场景YAML字段', '换成新场景的什么内容'], [
        ('scene.stage / scene.thermal_scene', '完整场景入口；热环境和天气配置'),
        ('drone.prim_path / body_path', '无人机根节点；实际运动的机体链接'),
        ('drone.camera_path / minimum_height', '机体下RGB Camera；世界Z下限，不是离地高度或避障'),
        ('sensors.sensor_config / thermal_config', '传感器定义；物体温度与材料绑定'),
        ('sensors.display_config / material_directory', '显示配置；本场景红外材料库'),
        ('sensors.width / height', '现场预览尺寸；不是离线256×256保存尺寸'),
        ('task.file', '本场景任务YAML，任务设计见附录D'),
        ('scene.collision_exclude_paths', '仅在需要时列出真实视觉特效路径；不能排除建筑、树干等实体'),
    ], [7.3, 10.1])
    code(r'''# my_scene.yaml必须已按实际场景改好 才执行检查
.\handoff\dataset-workflow-kit\check_environment.ps1 `
  -Config '.\configs\my_scene.yaml'
# 递归盘点USD依赖 下列stage路径换成真实入口
& 'D:\isaacsim\python.bat' -B scripts/dataset_asset_audit.py `
  --stage 'my_scene/flight.usda' --out 'validation/handoff/my_scene-assets.json' ''')
    para('软件检查和依赖扫描不证明活动场景完整。对unresolved与outside_project逐项判断，内置模块、非活动旧引用和活动缺失分开处理；实际打开后还要检查移动、回退、目标画面和回放。')
    para('工厂旧资产扫描曾发现远程无人机和贴图依赖，不能把旧工作包称为完整独立场景包。asset_delivery.csv记录来源、接收路径、清单和解决状态。')
    para('通用-Config不会自动按task.start_position设置出生位姿，也不会因复制YAML就增加-Scene选项。起点需在USD中准备，或由工程人员接入经过验证的固定起点入口。')

    page('附录C 相机红外与碰撞配置验收')
    heading('相机和传感器')
    para('先确认RGB Camera挂在运动机体下。运行时红外通常在body_path/Sensors/ThermalCamera；align_thermal_to_rgb开启时使用RGB视点。平移、升降和转向时，两路视点都应跟随。运行时修改不等于已保存回源USD。')
    para('本项目使用flir_boson_640_lwir.yaml及其光谱文件，波段7.5至13.5 μm，源模型名义640×512；当前离线256×256是独立渲染规格。两路同尺寸不等于像素严格同步或已标定对齐。')
    table(['需要固定的内容', '工程核对方法'], [
        ('视场和相机参数', '核对焦距、孔径、近远裁剪、外参与每帧有效内参'),
        ('热环境', '固定天气CSV、地点、start_utc与预设；区分合成设定与实测条件'),
        ('物体温度', '从已验证热配置复制，只用fixed、ambient、offset模式；温度单位为摄氏度'),
        ('红外材料', '核对目标Prim与子节点绑定、默认回退材料及光谱；RGB贴图不能替代'),
        ('显示方式', '固定温度窗、白热/黑热与色板，记录版本'),
    ], [4.3, 13.1])
    para('先使用真实加载器验证配置，再检查热绑定报告和两路画面。背景地面、植物、金属、玻璃和目标按实际类别配置；不要为了显眼随意增温，或把全部背景长期设成同一种材料。')
    heading('连续预览和按需快照')
    para('工厂、森林和城市固定任务用RGB驾驶与H红外快照；通用-Config默认连续配对预览。两者的目标参数通常为RGB约15 FPS、红外约2 FPS，实际受硬件和场景影响。轨迹降为每秒记录，不改变预览设置。')
    heading('新增场景的碰撞保护')
    para('人工录制默认启用static_sphere_undo_v2。保护器以无人机旋转包络球做控制步重叠与扫掠检查，覆盖可见静态网格和基本实体；碰撞体写入运行时会话，不保存源USD。新场景仍须验证起点净空、真实障碍、连续回退和继续录制。')
    para('动画实体或父节点运动、动态刚体尚不能按静态障碍处理，程序会拒绝；需要单独适配。森林配置只排除Flames和FireAtmosphere视觉特效，桶、树木、人物与地形仍参与碰撞。排除路径拼错会报错。')
    para('回退恢复无人机，不倒退整个世界。森林等离线动画需按source_simulation_time_s恢复原始时刻，不能用压缩后的分支时间替代。诊断可显式关闭保护，但不要在日常采集命令中关闭以绕过错误。')

    page('附录D 任务准备与现有工厂样例')
    para('任务负责人先给出目标和干扰目标、任务指令、起点与朝向、可通行范围、完成标准、参考图、版本及划分。场景和目标全集尚未完成时，先用这个样例学习方法，再逐项发布正式任务。')
    para('样例目标是灰色圆筒储罐，有蓝色护栏和红色外接管；在固定白热红外中，罐壁有局部近圆形较亮区域。弱光下结合轮廓和周围结构识别，不能只看颜色。')
    img = ROOT / 'handoff/dataset-workflow-kit/examples/factory-target-reference.png'
    doc.add_picture(str(img), width=Cm(10.7)); doc.paragraphs[-1].alignment = 1
    para('工厂目标参考图仅用于认识目标，不表示所有起点或正式任务均已验收。')
    para('英文指令样例  Find the gray cylindrical storage tank with blue safety railings and red external pipework. Identify it in RGB, inspect its wall in thermal imagery for a localized rounded warm patch, and stop where both the tank and the anomaly can be clearly inspected.')
    para('中文解释  找到指定储罐，在RGB中核对身份，用红外检查罐壁局部较亮区域，在能看清目标和异常的位置停住。任务中的合成温度和材料来源另存配置，不把它们写成从图片实测得到的事实。')
    code(r'''# 仅在该场景资产 相机和起点已由工程人员验收后试采
$runId = 'tutorial-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
.\scripts\record.ps1 -Config '.\configs\handoff_factory_example_scene.yaml' `
  -RunId $runId -HorizontalSpeed 3''')
    para('场景配置、任务文件为configs/handoff_factory_example_scene.yaml和handoff_factory_example_task.yaml；目标卡在handoff/dataset-workflow-kit/examples/。这个通用入口使用USD出生位姿、连续预览，升降默认0.5 m/s，不替代正式Episode的就绪检查。')
    para('任务表记录task_id、指定目标Prim、目标与背景描述、参考图、起点、完成规则及split。完成条件按该批任务单发布；本轮City的目标身份验收规则不能自动套到其他任务。')

    page('附录E 数据结构与回退来源说明')
    code(r'''validation/
  trajectory_runs/<RunId>/
    trajectory.jsonl           全部原始尝试
    run_metadata.json          任务 周期 精确T 回退事件
    training_trajectory.jsonl  有效T存在时生成的保留序列
    training_sequence_metadata.json
    config_snapshot/  mileage_quality.json
  collection_launches/<RunId>/   工厂 森林启动证据
  city-v2/launches/<RunId>.json   城市启动证据
  episodes/<RunId>/<处理批次>/
    pipeline_status.json  logs/
    replay/                    图片 数组 每帧及共享元数据
    dataset/                   index.json episode.json frames.jsonl
    derived_input/             也可能位于replay内
  annotations/<RunId>/<审核批次>/
    review.json  frame_review.csv''')
    table(['文件或字段', '读取方式'], [
        ('_rgb.png / _rgb.npy', 'RGB图与数组；当前默认256×256，数组可能有alpha通道'),
        ('_thermal.png / _display8.npy', '红外显示图，uint8；256×256，可能为RGBA'),
        ('_apparent_t.npy', '表观温度，float32、K；当前形状(256,256)，转摄氏度减273.15'),
        ('_radiance.npy / _dn16.npy', '辐亮度float32，单位与波段看元数据；DN为uint16，不是温度'),
        ('frame_*.json / common_metadata.json', '使用FrameMetadataReader合并共享字段并核对哈希'),
        ('dataset/frames.jsonl', '两路图片路径、观测时间、源帧号及source_event_ref'),
    ], [6.2, 11.2])
    para('世界位置单位米，Z向上，四元数顺序[w,x,y,z]。矩阵约定按项目读取器解释。索引可能含绝对路径；跨电脑保留原索引与哈希，另建旧根目录到新目录的映射，不静默猜另一批文件。')
    para('当前回放规格为paired_offline_v3，两路256×256直接渲染，保持水平视场并按方形画幅调整垂直视场及有效内参。历史capture_contract不改写；本次实际输出看replay_capture_contract、rgb_resolution、thermal_resolution。')
    para('collection_edit_events保存碰撞、回退锚点与原始帧。有效序列去掉撤销段、等待和回位跳变；simulation_time及索引observation_time_s为修正分支时间，source_simulation_time_s保留原始时钟。原始T独立映射成唯一STOP，未解决碰撞禁止导出。')
    para('交付完整批次和全部引用源，不能只按上面的示意树挑文件复制。修正分支里程是采样折线长度，不是连续运动真值。')

    page('附录F 新机验收和代码交付检查')
    heading('工程人员交出可用环境前')
    steps([
        '核对软件、GPU/驱动、路径和代码资源清单；在asset_delivery.csv登记资产。',
        '实际打开场景，核对无人机、相机、材质和出生点，再真实试飞。',
        '验证真实障碍、碰撞停止、回退与继续采样；不关闭保护来绕过错误。',
        '到目标按T并正常退出，回放核对256×256、1秒取样、精确终点和来源链。',
        '完成标注与独立复核；部署表写齐机器、版本、RunId、批次及报告位置。',
    ])
    para('工厂和森林旧readiness证据需按新代码重验，F01/R01可能仍为enabled:false；碰撞诊断不替代正式就绪检查，不手改哈希放行。每台接收电脑都要有自己的试采记录。')
    heading('用这些现有文件完成交接')
    table(['文件或入口', '用途'], [
        ('scripts/record.ps1 / process.ps1', '录制、回放与逐批处理'),
        ('scripts/collection_capture_defaults.py', '1秒记录/取样、256×256和契约版本的来源'),
        ('scripts/manual_collision_undo.py', '共用静态碰撞保护与回退；配合recording_undo.py'),
        ('scripts.dataset_handoff / create_review.ps1', '环境检查、文件核对、创建与核验审核表'),
        ('dataset_asset_audit.py / asset-audit.json', 'USD资源依赖盘点；旧报告需结合实际场景核对'),
        ('handoff/dataset-workflow-kit/templates/', '任务、采集日志、部署验收、发布、问题处理和资产交付CSV'),
        ('manifest.json / environment-baseline.json', '旧包清单与依赖参考；新版交付须重新生成并核对'),
    ], [8.3, 9.1])
    para('旧配套ZIP包含第三方红外源码，来源、版本及许可范围未核实前不要外发。它也不保证包含当前共用回退功能。交付方先确认可交付文件，再从当前工程准备代码、配置、依赖和完整资产，生成清单并做新机验收。')
    heading('模型训练前还需单独确认')
    para('当前bc_model_inputs.py是工厂前向RGB训练诊断接口；红外和人工审核字段尚未接入。若后续改为双模态训练、按审核结果自动筛选，或让模型学习可见性及飞行阶段，需另行实现读取、标签规则和验证。')
    para('现有prepare_bc_samples.py主要支持工厂正式Episode和有效现场T，City、森林和新场景需适配；1秒轨迹不能沿用旧0.5秒动作跨度。动作由有效分支的实际位姿差计算，保留两端时间和真实duration_s，不能直接把采样按键当动作。')
    para('候选标签yaw_local_future_pose_v1与执行动作factory_pose_action_v1/v2是不同接口；v1上限0.75秒，v2时长范围(0,1.25]秒。需显式匹配并验证后才进入训练。技术PASS、人工验收、严格像素同步和训练就绪分别记录。')
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    build()
