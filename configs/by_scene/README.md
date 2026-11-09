# 按场景查找配置

这里把同一场景的配置聚合在一个目录中。分类项是 Windows 快捷方式，双击打开原配置；原文件保留原路径，采集脚本和历史回放不改用这些快捷方式。

| 目录 | 内容 |
|---|---|
| [city](city/README.md) | City 的场景、红外、环境、材料、C01–C20 任务与采集清单 |
| [factory](factory/README.md) | 工厂 model02 各版本、巡检任务、采集计划与教学示例 |
| [forest](forest/README.md) | 森林场景、燃烧桶和行人目标、显示及材料 |
| [neighbourhood](neighbourhood/README.md) | 早期街区的昼夜场景与热配置 |
| [school](school/README.md) | 学校 VLN 场景与任务 |
| [shared](shared/README.md) | 共用材料、跨场景 pilot30 清单、无人机与通用示例 |

每个场景进一步按 `scenes`、`thermal`、`display`、`environment`、`tasks` 等用途分类。City、Factory、Forest 的 `current` 子目录额外汇集入口依赖。

`current` 不表示采集已通过门槛；具体运行仍以录制脚本、任务版本、运行快照和就绪检查为准。文件未列入 `current` 也不表示可以删除。

完整来源路径、来源 SHA256、分类及快捷方式位置见 `catalog.json`。本目录不参与红外渲染或任务选择。
