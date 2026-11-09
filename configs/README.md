# 配置导航

从 [按场景分类的目录](by_scene/README.md) 查找配置：

- [City 城市](by_scene/city/README.md)
- [Factory 工厂](by_scene/factory/README.md)
- [Forest 森林](by_scene/forest/README.md)
- [Neighbourhood 街区](by_scene/neighbourhood/README.md)
- [School 学校](by_scene/school/README.md)
- [共享材料、采集清单与示例](by_scene/shared/README.md)

`by_scene` 是分类导航，里面的 Windows 快捷方式打开原文件。现有配置的路径和内容均保留，录制、回放与历史哈希仍使用原文件。

在资源管理器中，打开 `by_scene` 下对应场景；`current` 汇集 City、工厂、森林的当前入口及其依赖。`current` 只表示配置入口，不代表任务、碰撞或数据质量已经验收。

## 文件类型

| 分类 | 用途 |
|---|---|
| scenes | USD 场景、无人机、相机与红外链路的运行入口 |
| thermal | 物体温度、红外材料和 Prim 路径绑定 |
| display | 温度到灰度的映射、极性与显示增强 |
| environment | 天气文件、时间、地点与大气参数 |
| tasks | 任务描述、目标、起点、朝向与任务索引 |
| targets | 目标注册与目标配置 |
| plans / collection | 采集计划、编号映射、数据划分和入口配方 |
| materials | 红外材料定义 |
| examples / experiments / backups | 示例、试验元数据和备份；未因分类而删除或判定无用 |

## 当前 City 采集

任务编号仍为 City v2 的 C01–C20，当前录制脚本加载 `city_200m_v3_scene.yaml`。场景入口和任务体系的版本号分别表示不同对象，不能根据文件名把两者同时切换。

`collection_profiles/city.yaml` 对应旧的 City 计划；`-Episode C01` 等编号走 `collection_episodes/city_v2.json` 和任务索引。继续使用现有采集命令。

## 维护

- 不把 `.lnk` 快捷方式作为 `--config` 参数；命令行使用原 YAML 路径。
- 修改配置时打开原文件。快捷方式不维护第二份 YAML，不会产生双份配置漂移。
- 新增配置后，在项目根目录运行 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/organize_config_scene_view.ps1` 更新分类。
- 加 `-CheckOnly` 可检查分类清单、全部快捷方式目标和来源哈希，不启动 Isaac Sim。
- 快捷方式包含本机绝对路径；迁移工程后重新生成。共享、打包与历史回放仍保留原配置及原有目录结构。
