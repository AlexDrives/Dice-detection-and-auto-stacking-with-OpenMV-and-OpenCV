# OpenMV 骰子识别与自动堆叠

这个项目是一套基于 OpenMV + OpenCV 的骰子识别和自动堆叠流程。OpenMV 主要负责采集图像，PC 端负责更重的视觉处理、像素坐标到机械臂世界坐标的映射，以及吸盘机械臂堆叠指令的生成和发送。

整体流程：

1. OpenMV 拍摄骰子图像
2. PC 端识别骰子中心和点数
3. 根据标定数据把像素坐标转换成世界坐标
4. 生成机械臂堆叠指令，并可选择通过串口发送

## 演示

![骰子自动堆叠演示](assets/6dice_stack_demo.gif)

上面的 GIF 已压缩并加速 2 倍，适合在 README 中快速预览。完整清晰版视频可以打开这里：

[查看原始演示视频](assets/6dice_stack.mp4)

## 仓库内容

核心脚本：

- `pc_openmv_capture_once.py`：触发 OpenMV 拍摄，并通过串口 REPL 把 JPG 图像拉回 PC。
- `pc_openmv_serial_pull.py`：从 OpenMV 串口拉取已经拍好的图像批次。
- `pc_openmv_disk_capture.py`：当 OpenMV 以 U 盘形式挂载时，直接从盘符复制图像。
- `pc_dice_detect.py`：基于 OpenCV 的本地图像骰子检测脚本。
- `yolo_dice_infer.py`：可选的 Ultralytics YOLO 骰子检测入口。
- `camera_undistort.py`：相机内参读取和图像去畸变工具。
- `validate_calibration_models.py`：验证像素坐标到世界坐标的标定模型。
- `dice_stack_pipeline.py`：从检测结果到堆叠指令的完整流水线。
- `dice_stack_commands.py`：根据世界坐标生成堆叠命令。
- `dice_restore_commands.py`：根据世界坐标生成还原命令。

配置和标定文件：

- `dice_stack_config.yaml`：简单堆叠/还原脚本使用的机械臂参数。
- `dice_stack_pipeline_config.yaml`：完整流水线使用的参数。
- `calibrations/`：相机内参、像素到世界坐标映射等标定文件。
- `coords/semi_auto_calibration_points.csv`：用于标定验证的人工确认点。
- `相机标定/`：棋盘格生成和 OpenCV 相机标定工具。

本地采集和调试输出不会上传到 Git，例如 `captures/`、`results/`、`logs/`、原始标定图片、Jupyter Notebook、临时视频和第三方解包文件。

## 环境要求

推荐使用 Python 3.10 或更高版本。

安装基础依赖：

```powershell
pip install -r requirements.txt
```

如果只使用 OpenCV 检测流程，可以不训练 YOLO；`ultralytics` 只在运行 `yolo_dice_infer.py` 时需要。使用 OpenMV 串口工具时，请确认开发板已经出现在系统串口中，例如 `COM17`。

## 快速开始

从 OpenMV 采集一批新图像：

```powershell
python pc_openmv_capture_once.py --port COM17 --expected 6
```

图像会保存到：

```text
captures/openmv_YYYYMMDD_HHMMSS/
```

对某个采集目录运行 OpenCV 骰子检测：

```powershell
python pc_dice_detect.py --input captures/openmv_YYYYMMDD_HHMMSS --body-threshold 180 --save-binary
```

如果光照变化明显，可以对比 `170`、`175`、`180`、`185` 等阈值，并查看保存的二值图。

根据世界坐标 CSV 或 JSON 生成堆叠指令：

```powershell
python dice_stack_commands.py --coords-file coords/example_world_coords.csv
```

使用完整流水线处理已经生成的检测结果：

```powershell
python dice_stack_pipeline.py --input results/dice_detect/frame_01.json
```

真正连接机械臂发送指令前，请先检查 `dice_stack_pipeline_config.yaml`。其中的运动范围、拾取高度、堆叠目标点、吸盘延时和串口指令名都需要与你的机械臂控制器一致。

## 相机标定

`相机标定/` 目录中提供了棋盘格生成和相机内参标定工具：

```powershell
cd 相机标定
python generate_checkerboard.py
python calibrate_camera.py --annotate --undistort-sample
```

生成的相机内参可以复制到 `calibrations/`，并在检测时传入：

```powershell
python pc_dice_detect.py --input captures/openmv_YYYYMMDD_HHMMSS --intrinsics calibrations/camera_intrinsics_paper_9p63_simple.json
```

## YOLO 检测方案

如果已经训练好骰子检测模型，可以运行：

```powershell
python yolo_dice_infer.py --model best.pt --source 0 --show
```

推荐类别名为 `d1`、`d2`、`d3`、`d4`、`d5`、`d6`。脚本会把类别名映射成骰子点数，并输出每一帧的检测中心。

## 注意事项

- 当前参数针对特定 OpenMV 相机、光照、标定板高度和机械臂坐标系调过。
- 更换相机位置、镜头、放置平面高度或机械结构后，需要重新标定。
- 在真正控制硬件前，建议先不加 `--send`，只检查生成的指令是否合理。
- 机械臂运动有风险，请先低速、空载、单步验证。

## 许可证

本项目使用 MIT License，详见 `LICENSE`。
