# V1.6 字体还原实现与验收记录

> 历史记录：本记录描述的 ResNet 回退和传统图像规则路线已于 2026-10-22 按用户要求废止。当前代码只使用本机字体栅格渲染拟合，失败时中文回退 SimHei、英文回退 Times New Roman；下文仅用于说明旧版本历史，不代表当前实现。

日期：2026-10-21

## 实现结果

- 接入传统字形匹配、有效字形计数与多样性拒识、候选字体面限定的整框 `render_fit`、P0 影子记录、带独立验证集/留出集证据的门控标定，以及 CV 拒识后的现有 ResNet + `render_fit` 回退。
- 增强路径只能通过 `--font-match-enhancement` 显式启用；缺少标定配置、校准证据或独立留出集时自动保持影子/回退。支持按 layout element ID 强制回滚。
- 字体清单列出 21 个目标家族、真实字体面索引、cmap、字体面 ID、PPT 字体名、来源与许可声明状态。字体路径只引用本机/用户配置文件；本次没有下载、复制、分发或嵌入字体，也没有改动 OCR/ResNet 模型、权重或预处理。
- `render_fit` 诊断将拟合总分、先验惩罚、prior-free 视觉损失分开；候选限定和 TTC/OTC face index 保持在结果中。

## 验证证据

- `python -m compileall -q scripts tests`：通过。
- `python -m unittest discover -s tests`：60 项通过。
- 隔离的单页完整 pipeline smoke：63 个文本对象、74 个图像对象；63 个文本对象均产生了传统 CV 候选，0 个运行时异常；63 项均为 `shadow_only`，基线未被覆盖。
- 单页 prior-free 视觉比较有 63 对样本，平均损失差（基线减增强）为 `0.000656`，33 项改善、30 项变差。该样例没有人工字体真值，结果只作机制 smoke，不能视为识别精度或稳定收益证明。
- 本机字体目录扫描：21 个家族均有清单项；12 个家族解析出 20 个字体面，9 个家族缺少本机字体文件：思源黑体、思源宋体、OPPO Sans、霞鹜文楷、得意黑、优设标题黑、庞门正道标题体、站酷快乐体、方正兰亭黑。目标 PowerPoint 字体可用性仍标为 `unknown`。
- Microsoft PowerPoint Office16 实际打开并导出了一页 21 家族字体检查 PPTX，PNG 渲染成功：`output/font_match_v1_6_powerpoint_smoke_20261021/font_family_render_check.pptx`、`幻灯片1.PNG`。这确认了本机 PPTX 打开/导出链路，但缺失字体可能被 PowerPoint 替换；PowerPoint COM 字体集合返回 0 项，因此逐家族目标端可用性仍是 `unknown`，也没有测量 `ppt_render_loss_delta`。
- 影子报告：`output/font_match_v1_6_smoke_20261021/debug/font_match_shadow/page_01.json`、`font_catalog.json`、`font_match_p0_ab.json`。
- PowerPoint 环境探测明细：`output/font_match_v1_6_powerpoint_smoke_20261021/font_availability.json`。

## 2026-10-22 本机字体资源更新

- 用户授权后，从官方发布渠道将 5 个此前缺失的字体家族安装到用户级配置目录 `%LOCALAPPDATA%\image2ppt\fonts`，并设置用户环境变量 `IMAGE2PPT_FONT_DIRS` 指向该目录。字体文件未加入仓库，也未嵌入 PPT。
- 运行时目录扫描现识别 17/21 个目标家族、40 个可用字体面记录；本次新增 20 个字体文件/记录：思源黑体 7、思源宋体 7、霞鹜文楷 3、得意黑 2、OPPO Sans 1。扫描无解析错误，5 个新增家族均为 `matching_availability=available`。OPPO Sans 4.0 是含 `wght` 轴的可变字体，带 5 个命名字重实例；当前目录扫描器只登记 1 个字体面记录，尚未把可变字重展开为独立候选。
- 用户目录的 `license_manifest.json` 为新增字体记录来源、许可、SHA-256、本地匹配许可及再分发状态；OPPO Sans 4.0 的原始许可声明保存在字体目录，并在仓库 `THIRD_PARTY_NOTICES.md` 和 `README.md` 添加了使用提示。
- 尚缺 4 个家族：优设标题黑、庞门正道标题体、站酷快乐体、方正兰亭黑。前 3 个本轮未能从可访问且可核验的发布方官方下载入口取得字体文件；方正兰亭黑需字体授权。未使用第三方镜像或购买字体。
- 本次只验证了运行时字体目录扫描和许可字段，没有重跑 PPT 转换或 V1.6 字体准确率回归；目标 PowerPoint 端可用性仍为 `unknown`，V1.6 生产启用条件仍未满足。

## 尚未满足的 V1.6 验收条件

1. 本次安装后仍缺少 4 个目标家族的真实字体资产或授权提供路径，因此 21/21 字体家族覆盖条件未通过。
2. 尚无冻结的已知字体真值集及独立留出集，不能给 Top-1/margin/质量门槛定标；运行时保持 `shadow_only`，不能把候选结果用于生产 PPT 字体覆盖。
3. 尚无同一 PowerPoint 环境下、仅待测文本样式不同的成对 PPTX 渲染结果；`ppt_render_loss_delta` 未测量。

因此，本次完成的是 V1.6 所需实现与自动化机制；**V1.6 的 P0 验收/启用退出条件尚未完成**。上述数据、资产和目标环境渲染证据补齐前，默认 PPT 导出仍走原有 ResNet + `render_fit` 基线。

## 2026-10-22 字体候选清单调整

- 按用户要求，Alibaba PuHuiTi 2.0 已从当前目标字体清单和本机渲染候选中移除；当前清单为 20 个目标家族。旧格式布局若仍引用该名称，导出时中文改用 SimHei，纯英文改用 Times New Roman。
- 以上 V1.6 数量是当时 21 家族清单下的历史扫描记录，不代表当前 20 家族清单的最新可用状态。运行时字体清单已同步标注 20 个目标家族。
