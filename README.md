# 一种基于光流约束的双目夜间驾驶眩光抑制方法及系统

用同步拍摄的正常曝光图和减光图抑制夜间车灯、路灯及湿路反光。正常曝光图保留暗部，可靠对齐的减光图补充强光区域。支持双图、左右拼接图像或视频，以及同步双目拼接相机。

下图从左到右为合成场景的正常曝光、对齐减光和最终输出。

![合成场景的正常曝光、对齐减光和融合结果](assets/synthetic-comparison.png)

## 快速运行

需要 Python 3.10–3.13。CPU 即可运行，无模型权重或外部数据下载。建议使用虚拟环境。

```bash
python -m venv .venv
```

Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1`，macOS 或 Linux 用 `source .venv/bin/activate` 激活后运行：

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_pipeline
python make_demo.py --out samples/synthetic
python run.py image --packed samples/synthetic/packed.png --out outputs/synthetic
```

Windows 也可以在安装依赖后双击 `run_demo.cmd`，运行测试和合成样例。

## 使用自己的双目素材

```bash
python run.py image --left left.png --right right.png --out outputs/image
python run.py video --packed stereo.mp4 --out outputs/video --video-out outputs/video/fused.mp4
python run.py video --left left.mp4 --right right.mp4 --out outputs/pair --video-out outputs/pair/fused.mp4
```

- 默认左路是正常曝光，右路是减光。顺序相反时加 `--normal-side right`。双文件按帧序号配对，须在采集时同步并保持帧率、帧数一致。
- 左右拼接帧默认等宽拆分。有边缘留白时可传 `--crop-config crop.json`，文件包含 `left_crop` 和 `right_crop`，各自使用像素字段 `x`、`y`、`width`、`height`。
- 未校正的输入可通过 `--calibration` 传入自身设备的 JSON 或 NPZ。JSON 包含 `[宽, 高]` 格式的 `image_size`、`left_camera_matrix`、`left_distortion`、`right_camera_matrix`、`right_distortion`、双目旋转 `R` 和平移 `T`。NPZ 预计算映射表使用 `map1x`、`map1y`、`map2x`、`map2y`。
- 已校正图像若有黑边，可传 `--mask left-valid.png right-valid.png`；白色为有效区域。视频输出保留帧率，没有音轨。

`python run.py camera --help` 查看拼接相机预览和原始录像参数。硬件须自行保证同步，实体相机入口未在公开样例中验证。

## 与专利方案的对应关系

| 步骤 | 代码 | 实现 |
| --- | --- | --- |
| S101 同步图像获取 | `inputs.py`、`run.py` | 拆分双目拼接帧，按标定查找表双线性校正。输入同步由采集设备保证。 |
| S102 光流估计与对齐 | `GlareSuppressor.process` | 按非饱和灰度统计归一化减光路，估计多尺度双向 DIS 光流，以正常曝光坐标重采样减光图。 |
| S103 光流可靠性 | `flow_reliability`、`structure_confidence` | 分别计算双线性采样有效区、局部归一化相关置信度、双向往返误差及遮挡掩膜。 |
| S104 强光门控 | `glare_masks` | 双图亮度判断灯芯，膨胀得到候选区，以可靠非饱和像素的亮度比中位数作增益，再以补偿后亮度差判别光晕。 |
| S105 增强与权重 | `GlareSuppressor.enhance`、`fusion_weights` | 中央区亮度差经时间平滑和滞回控制中值滤波及 CLAHE。低置信度减小减光贡献，遮挡／无效处减光权重为零，辅助增强受占比上限约束。 |
| S106 多尺度融合 | `pyramid_fuse` | 同帧三路图像经拉普拉斯金字塔、权重高斯金字塔、逐层归一化和重建得到融合结果。 |

用 `configs/patent.json` 运行 S101–S106 复现模式。该配置使用线性亮度归一化、文档示例中的 9×9 相关窗口、0.65 高置信度门限、1.5 像素往返误差门限和 21 像素候选区膨胀半径。低置信度采用连续的权重衰减。

```bash
python run.py image --packed samples/synthetic/packed.png --config configs/patent.json --out outputs/patent
```

默认模式在 S106 后额外执行高光压缩，并限制局部亮度突降，改善显示效果。复现配置关闭这一步，`fused.png` 即 S106 的输出。

`--out` 保存处理首帧的诊断。`raw_fused.png` 为显示调整前的融合，`fused.png` 为最终图像。`fields.npz` 含光流、置信度和融合权重；`report.json` 含参数、亮度增益和高光统计。

使用 `--config config.json` 覆盖默认参数。例如 `{"highlight_rolloff": 0.38, "fusion_drop_limit": 32.0}` 会加强高光压缩并限制减光落差。将 `highlight_rolloff` 设为 0 可以关闭显示调整。

已经饱和且在减光图中找不到可靠对应位置的纹理无法恢复。本项目使用可控的合成场景测试，离线结果会随曝光差、双目同步和校正质量变化。
