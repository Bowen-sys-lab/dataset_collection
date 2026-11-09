# city_200m 本地材质

本目录只补旧版 irsim（isaac-thermal-camera-simulation, 563a51b）缺的两条材质，
内容逐字节抄自作者使用的 irsim 15015cd，未改数值：

- `polyester_clothing.yaml` — 人物外套
- `shoe_rubber.yaml` — 人物鞋

作者的 `concrete_weathered` 没有抄：它依赖 SLUM 实测谱线（旧版 data/ 里没有），
转换时归到同类 `concrete`，见 scene_intake/convert_city_200m.py 的 MATERIAL_MAP。
