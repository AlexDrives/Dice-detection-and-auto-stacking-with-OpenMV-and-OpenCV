# openMV Workspace Structure

## 常用入口

- `pc_openmv_capture_once.py`: 触发 OpenMV 拍摄并把图像拉回本机。
- `pc_openmv_serial_pull.py`: 从 OpenMV 串口 REPL 拉取已经拍好的图像。
- `pc_openmv_disk_capture.py`: Windows 能刷新盘符时，直接从 OpenMV U 盘目录复制图像。
- `pc_dice_detect.py`: PC 端 OpenCV 骰子检测和中心提取。
- `yolo_dice_infer.py`: PC 端 YOLO 骰子检测入口。
- `dice_stack_pipeline.py`: 检测、坐标转换、堆叠指令生成的主流程。
- `dice_stack_commands.py`: 根据世界坐标生成或发送机械臂堆叠命令。
- `dice_restore_commands.py`: 根据世界坐标生成或发送机械臂还原命令。
- `dice_stack_config.yaml`: 简单堆叠/还原脚本的机械臂参数。
- `dice_stack_pipeline_config.yaml`: 完整流水线参数。

## 目录

- `calibrations/`: 相机内参和像素到世界坐标的标定文件。
- `coords/`: 本地坐标记录；开源时只保留 curated calibration CSV。
- `captures/`: OpenMV 拉回的原始图像批次，本地输出，不上传。
- `results/`: 检测、标定验证和调试结果，本地输出，不上传。
- `logs/`: 运行日志，本地输出，不上传。
- `docs/`: 参考资料，可能包含第三方材料，默认不上传 PDF。
- `相机标定/`: 棋盘格生成和 OpenCV 相机标定工具。

## 主线

当前主线是：

```text
OpenMV 拍图 -> PC 识别 -> 标定映射 -> 串口/文件输出堆叠命令
```
