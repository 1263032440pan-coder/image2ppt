# VLM Inverse Graphics Lab · v0.5

从 **观察 → 理解 → 设计 → Geometry / Color / Detail IR → SVG → 渲染 → 分区域评价 → 局部修复** 构建一条可运行的静态二维逆向图形学管线。

> **重要：** v0.5 默认采用参考图驱动的 **SVG 连续渐变色场 + 几何轮廓 + 独立细节路径**。不再将 SLIC 超像素块逐个绘成可见多边形，所以没有 v0.3 的格子。它仍是源图引导的可编辑矢量近似，**不是** VLM 独立创作的语义绘图程序，SVG 不嵌入源位图。

## 快速启动

- Windows：双击 `start-windows.bat`。
- Linux / macOS：`bash start-unix.sh`。
- 访问 <http://127.0.0.1:8000>。
- 首次运行自动创建 `.venv` 并安装依赖。Chromium 安装失败可直接安装 `cairosvg` 作兼容回退（某些 SVG 特性可能不同）。
- 复制 `.env.example` 为 `.env`，需要真实 VLM 调用时设置 `MOMA_API_KEY`。**切勿提交 `.env` 或把 API Key 放入代码和浏览器。**
- 选择内置锦鲤示例并勾选「演示模式」可以离线运行。演示使用 `examples/demo_scene.json` 中预定义的对象，并**不调用 VLM**。

真实 VLM 模式使用用户指定的 Anthropic Messages 风格端点，默认：

```env
MOMA_API_URL=https://vip.moma.cmecloud.cn/v1/messages
MOMA_MODEL=deepseek-v4.1-flash-1017
MOMA_IMAGE_FORMAT=auto
MOMA_API_KEY=
```

`MOMA_IMAGE_FORMAT=auto` 会先试 Anthropic `source.base64` 图片格式，仅在格式相关 HTTP 400/415/422 时尝试 OpenAI `image_url` 形式。**尚未完成该第三方网关的在线联调，不保证它允许当前模型的多模态请求。**

## v0.5 修复内容与早期版本对比

| 方面 | v0.3 | v0.5 |
| --- | --- | --- |
| Color IR | SLIC 色块多边形 + 1.3px 描边，形成方格 | 参考图测量的窄带连续 SVG 渐变，无色块描边 |
| 几何 | GrabCut + 轮廓/候选部件 | 保留 v0.3 的几何模块 |
| 细节 | Canny 路径可能意外连接不相邻的点 | 超长连接断开，减少异常斜线 |
| 返工 | 叠加细分多边形 | 仅加密误差区域的渐变采样点 |
| 评分 | ΔE00 + 边缘 F1 | 保留原指标，增加反格子回归测试 |
| 稳定性 | 隔离任务/渲染进程 | 沿用隔离机制 |

## v0.5 高保真清晰度修复（2026-11-17）

- v0.4 的颜色先缩小至 `640×640`，再双边滤波、高斯模糊，放大成 `1000×1000`；细节层则先缩至 `800×800`。这是明显柔化的原因。
- v0.5 在 `1000×1000` 输出分辨率直接采样（原图较大时只缩放一次），`COLOR_BAND_HEIGHT=1`、`COLOR_SAMPLE_STEP=6`；真正的图像颜色边缘附近增加插值停靠点。
- 增加 Canny 原图分辨率追踪和更精细的线条路径。由于原图墨线已包含在新的 Color IR，细节矢量路径采用 `DETAIL_OVERLAY_STRENGTH=0.20` 的弱叠加，避免二次描边制造脏乱。设为 `0` 可仅看颜色层与 Geometry IR；`1` 显示完整 Detail IR 强度。
- 格子没有恢复，因为这仍然是 SVG 横向渐变，不绘制 SLIC 多边形或多边形描边。代价是 SVG 包含大量渐变停靠点，绘制性能与源图复杂度相关。
- **限制**：精细还原的主要来源仍是直接沿原图采样，不能当作 VLM 独立绘制复杂鱼鳞与枝叶的证明。真正语义可编辑的部件级矢量生成仍需后续开发。

## 执行流程

1. **观察/理解**：真实模型视觉识别对象、位置、遮挡；演示模式读取固定场景。
2. **设计**：只接受有边界、可信的 JSON Scene IR，禁止模型直接输出可执行代码或任意 SVG。
3. **Geometry IR** (`app/geometry.py`)：实例 ROI + GrabCut + foreground/background 点；将源图轮廓转换为路径；通过 Hough 提取部件候选线段。失败保留对象元信息。
4. **Color IR** (`app/color.py`)：从参考图测量空间 RGB 颜色，不再缩到 640px 或做高斯模糊，直接按输出分辨率生成 1px 高的 SVG 渐变扫描带。没有 SLIC 可见多边形；每条渐变带与颜色停止点可编辑。
5. **Detail IR** (`app/detail.py`)：全场景 Canny 矢量化；对折返轮廓降采样造成的异常长连接进行拆分，避免画面出现斜穿背景的假线条。
6. **SVG Renderer** (`app/graphics.py`)：原生 `<linearGradient>` 和无描边 `<rect>` 组成连续颜色底层，叠加可编辑几何与细节。旧参数化对象引擎仍能在关闭 Color IR 后使用。
7. **Review** (`app/region_metrics.py`, `app/repair.py`)：分区 ΔE00 和边缘 F1；真实模型可以修正 Scene JSON，局部颜色修复通过加密渐变采样完成，不增加可见网格。

### 产物

工作目录 `.runtime/{job_id}/`（被 git 忽略）：

- `observation.json`, `scene.json`, `geometry.json`, `color.json`, `detail.json`
- `preview.svg`（原生 `<linearGradient>`、`<rect>`、`<path>`，不含 `<image>` 位图）；`render.png`
- `region_metrics.json`, `color_heatmap.png`, `report.json`
- 各轮 `iter_*.svg`, `iter_*.png`, `iter_*_color.json` 等记录

新增 API：

- `POST /api/jobs`：表单字段 `image`、`iterations`、`demo`、`geometry_enabled`、`color_enabled`、`detail_enabled`。
- `GET /api/jobs/{id}`：状态、进度和最终报告。
- `GET /api/jobs/{id}/files/{filename}`：仅允许下载白名单产物。
- 本地单用户 MVP **同时最多运行一个任务**；任务在子进程处理，15 分钟超时，状态仍由内存管理，重启会清空任务列表。

## 如何评价

基于随包示例浮世绘，同一套 `region_comparison()` 指标的离线演示：

| 指标 | v0.4 历史演示 | v0.5 当前端到端演示 |
| --- | ---: | ---: |
| 平均 ΔE00（越低越好） | 3.690 | **2.666** |
| 边缘 F1（越高越好） | 0.9767 | **0.9925** |

v0.5 的数值来自项目同一张锦鲤示例、演示模式的独立进程渲染（CairoSVG 回退）。与 v0.4 的数值是历史演示结果的对比，未进行其他数据集泛化验证。`report.json` 有完整记录。改进主要来自直接采样原图的颜色和线条，不代表 VLM 自动绘图能力同比提升。对任意图像的效果不能外推。

**可见代价：** 横向渐变带近似连续颜色，局部颜色可能略显柔和；虽然不再有 v0.3 的格子，仍可能有少量色带、曲线误差或对象遮挡不准确。当前也不是逐对象语义绘图程序。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python scripts/run_tests_isolated.py
```

图像处理测试涉及 OpenCV / CairoSVG / scikit-image 的原生状态，建议按模块运行在**独立 Python 进程**，CI 分步骤运行每个测试模块。v0.5 已验证清晰度/反网格测试、geometry/graphics/schema/model/pipeline 测试，以及单次演示完整流程；`test_real_flow_mock.py` 的连续执行出现超时，尚未解决，真实第三方模型 API 仍未在线验证。若整套测试在你的机器上卡住，可逐个运行 `python -m pytest -q tests/test_*.py`（一次指定一个文件）。

Windows + 旧 CPU 可调高 `.env` 中的 `COLOR_BAND_HEIGHT=3`、`COLOR_SAMPLE_STEP=15`，并降低 `DETAIL_MAX_PATHS=1500`；数值越大，颜色渐变越稀疏，速度更快但可能降低精度。建议默认 `COLOR_BAND_HEIGHT=1`、`COLOR_SAMPLE_STEP=6`，并可用 `DETAIL_OVERLAY_STRENGTH=0.20` 调整附加线条的强度。`COLOR_SEGMENTS` 已废弃。

## 已知限制与安全

1. **不是 SAM**：目前基于传统 OpenCV 的 GrabCut，复杂遮挡、同色背景可能失败。`parts` 是带标签的 Hough 候选线段，不是可靠的语义化部件掩码；独立编辑一个物体的全部内部色块还需未来的 Mask 精确关联。
2. 默认 Color IR 是**原图引导的可编辑渐变矢量近似**，不是每个物体的语义颜色模型。关闭 Color IR 会回退程序化模板，通常降低精度。
3. SVG 可能包含数十万条渐变停靠点和大量细节路径，文件较大；当前没有高层语义部件颜色建模、重光照、反射生成或高级 Shader。
4. 对象分数使用可见 GrabCut mask（估计值）；失败时回退 BBox，不可把它当作分割 ground truth。
5. 不适合部署在公网：没有身份验证，未设计多用户队列或存储清理。只能绑定到 `127.0.0.1`。
6. 真实模型的第三方 API Key 必须只保存在本机 `.env`，绝不写入前端或提交 GitHub。

## GitHub 发布

当前 GitHub 集成没有提供新仓库创建权限，项目以 ZIP 交付。可在本地使用 GitHub CLI：

```powershell
git init -b main
git add .
git commit -m "fix: preserve fine source detail without grid artefacts v0.5"
gh repo create 1263032440pan-coder/vlm-inverse-graphics-mvp --private --source=. --remote=origin --push
```

确保仓库 `private`，确认 `.env` 未被提交。
