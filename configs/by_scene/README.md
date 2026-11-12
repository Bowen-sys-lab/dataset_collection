# 按场景查找配置

这里把同一场景的配置聚合在一个目录中。分类项是 Windows 快捷方式，双击打开原配置；原文件保留原路径，采集脚本和历史回放不改用这些快捷方式。

| 目录 | 内容 |
|---|---|
| [city](city/README.md) | City 的场景、红外、环境、材料、任务与采集清单 |
| [neighbourhood](neighbourhood/README.md) | 早期街区的昼夜场景与热配置 |
| [school](school/README.md) | 学校 VLN 场景与任务 |
| [shared](shared/README.md) | 共用材料、跨场景 pilot30 清单、无人机与通用示例 |

City 配置进一步按 `scenes`、`thermal`、`display`、`environment`、`tasks` 等用途分类。

具体运行仍以录制脚本、任务版本、运行快照和就绪检查为准。本目录只用于阅读配置，不参与红外渲染或任务选择。
