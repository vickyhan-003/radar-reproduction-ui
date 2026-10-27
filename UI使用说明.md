# RADAR 复现 UI 使用说明

## 启动

在 PowerShell 中运行：

```powershell
Set-Location E:\radar_test\damo-radar
.\start_ui.ps1
```

浏览器会自动打开本地页面。若未自动打开，请访问终端显示的 `Local URL`。

## 操作流程

1. 在左侧选择“官方示例”或上传一个 `.nii.gz` CT 文件。
2. 8 GB 显存保持“低显存 · 64 × 160 × 224”。
3. 点击“开始预测”。
4. 查看概率图、完整结果表，并下载原始 CSV。

## 复现说明

- 模型：`ckpt/checkpoint_radar_pretrain.pth`
- 文本编码器：`ckpt/bert-base-chinese/`
- 官方样例：`data/demo_cases/AC423ccbe.nii.gz`
- 官方命令行结果：`results/RADAR_infer_results_demo.csv`
- UI 临时输入和结果：`.ui_runtime/`

该界面与命令行使用同一份 `inference_demo.py`，低显存模式通过更小的滑窗、FP16 前向和 CPU 分割拼接适配 8 GB 显存，不修改模型权重。

仅供课程复现与科研验证，不用于临床诊断。
