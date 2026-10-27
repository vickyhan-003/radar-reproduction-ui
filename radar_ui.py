from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
INFERENCE_DIR = ROOT / "RADAR_inference"
RESULTS_DIR = ROOT / "results"
RUNTIME_DIR = ROOT / ".ui_runtime"
SAMPLE_DIR = ROOT / "data" / "demo_cases"
MODEL_PATH = ROOT / "ckpt" / "checkpoint_radar_pretrain.pth"
BERT_PATH = ROOT / "ckpt" / "bert-base-chinese" / "pytorch_model.bin"


st.set_page_config(
    page_title="RADAR 复现工作台",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --radar-primary: #6d28d9;
      --radar-accent: #0e7490;
      --radar-ink: #1e1b4b;
      --radar-muted: #475569;
      --radar-border: #ddd6fe;
      --radar-surface: #ffffff;
    }
    [data-testid="stAppViewContainer"] {
      background: linear-gradient(145deg, #faf5ff 0%, #f8fafc 46%, #ecfeff 100%);
      color: var(--radar-ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
      background: rgba(255,255,255,.88);
      border-right: 1px solid var(--radar-border);
    }
    .block-container { max-width: 1320px; padding-top: 2rem; }
    .radar-kicker {
      color: var(--radar-primary); font-size: .78rem; font-weight: 800;
      letter-spacing: .14em; text-transform: uppercase; margin-bottom: .5rem;
    }
    .radar-title { font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.04; margin: 0; }
    .radar-subtitle { color: var(--radar-muted); font-size: 1.05rem; max-width: 720px; }
    .status-card {
      background: rgba(255,255,255,.84); border: 1px solid var(--radar-border);
      border-radius: 16px; padding: 14px 16px; min-height: 92px;
      box-shadow: 0 10px 30px rgba(76,29,149,.06);
    }
    .status-label { color: var(--radar-muted); font-size: .78rem; font-weight: 700; }
    .status-value { color: var(--radar-ink); font-size: 1rem; font-weight: 800; margin-top: .35rem; }
    .step-label { color: var(--radar-primary); font-size: .78rem; font-weight: 800; letter-spacing: .08em; }
    .research-note {
      border-left: 3px solid var(--radar-accent); background: rgba(236,254,255,.8);
      padding: 12px 14px; border-radius: 0 10px 10px 0; color: #164e63;
    }
    div.stButton > button, div.stDownloadButton > button {
      min-height: 44px; border-radius: 10px; font-weight: 750;
      transition: box-shadow .18s ease, border-color .18s ease;
    }
    div.stButton > button:focus-visible, div.stDownloadButton > button:focus-visible {
      outline: 3px solid rgba(109,40,217,.3); outline-offset: 2px;
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def status_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="status-card"><div class="status-label">{label}</div>'
        f'<div class="status-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def preview_slice(path: str) -> np.ndarray:
    volume = np.asarray(nib.load(path).dataobj)
    if volume.ndim == 4:
        volume = volume[..., 0]
    image = np.rot90(volume[:, :, volume.shape[2] // 2])
    image = np.nan_to_num(image, nan=-150.0, posinf=250.0, neginf=-150.0)
    image = np.clip(image, -150, 250)
    return (image + 150) / 400


def safe_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name)
    return clean if clean.endswith(".nii.gz") else f"{clean}.nii.gz"


def save_upload(uploaded_file) -> Path:
    case_dir = RUNTIME_DIR / "cases" / uuid.uuid4().hex[:10]
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / safe_name(uploaded_file.name)
    path.write_bytes(uploaded_file.getbuffer())
    return path


def run_inference(case_dir: Path, roi_size: str) -> tuple[Path, str]:
    run_id = uuid.uuid4().hex[:10]
    output_dir = RUNTIME_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RADAR_ROI_SIZE"] = roi_size
    command = [
        sys.executable,
        "-u",
        "inference_demo.py",
        "--img_dir",
        str(case_dir),
        "--save_dir",
        str(output_dir),
        "--save_tag",
        run_id,
    ]
    completed = subprocess.run(
        command,
        cwd=INFERENCE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    log = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    result_path = output_dir / f"RADAR_infer_results_{run_id}.csv"
    if completed.returncode != 0 or not result_path.exists():
        raise RuntimeError(log or f"推理进程退出码：{completed.returncode}")
    return result_path, log


def to_long_results(frame: pd.DataFrame) -> pd.DataFrame:
    row = frame.iloc[0]
    records = []
    for column in frame.columns[1:]:
        value = pd.to_numeric(row[column], errors="coerce")
        if pd.isna(value):
            continue
        chinese = column.split(" (")[0]
        english_match = re.search(r"\((.+)\)$", column)
        organ, _, finding = chinese.partition("_")
        records.append(
            {
                "器官": organ,
                "影像征象": finding or chinese,
                "英文名称": english_match.group(1) if english_match else "",
                "阳性概率": float(value),
            }
        )
    return pd.DataFrame(records).sort_values("阳性概率", ascending=False, ignore_index=True)


st.markdown('<div class="radar-kicker">Reproduction Console / 复现工作台</div>', unsafe_allow_html=True)
st.markdown('<h1 class="radar-title">RADAR 腹部 CT 推理</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="radar-subtitle">使用论文公开权重完成 3D CT 影像征象预测。'
    '界面记录从输入病例、模型推理到结果导出的完整复现链路。</p>',
    unsafe_allow_html=True,
)

try:
    import torch

    gpu_ready = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ready else "未检测到 CUDA"
    torch_version = torch.__version__
except Exception:
    gpu_ready, gpu_name, torch_version = False, "Torch 加载失败", "未知"

checks = {
    "模型权重": MODEL_PATH.exists(),
    "BERT 权重": BERT_PATH.exists(),
    "CUDA": gpu_ready,
}
ready = all(checks.values())

summary_cols = st.columns(4)
with summary_cols[0]:
    status_card("复现状态", "可以运行" if ready else "需要检查")
with summary_cols[1]:
    status_card("计算设备", gpu_name)
with summary_cols[2]:
    status_card("PyTorch", torch_version)
with summary_cols[3]:
    status_card("推理模式", "FP16 · 低显存滑窗")

with st.sidebar:
    st.subheader("运行检查")
    for label, ok in checks.items():
        st.markdown(f"{'通过' if ok else '缺失'} · {label}")
    st.divider()
    st.subheader("推理设置")
    source = st.radio("病例来源", ["官方示例", "上传 NIfTI"], horizontal=False)
    roi_label = st.selectbox(
        "滑窗尺寸",
        ["低显存 · 64 × 160 × 224", "论文默认 · 96 × 256 × 384"],
        help="8 GB 显存请选择低显存模式；论文默认尺寸需要更大的显存。",
    )
    roi_size = "64,160,224" if roi_label.startswith("低显存") else "96,256,384"
    threshold = st.slider("结果展示阈值", 0.0, 1.0, 0.50, 0.05)
    st.divider()
    st.caption("仅供课程复现与科研验证，不用于临床诊断。")

st.divider()
left, right = st.columns([1.05, 1], gap="large")

case_path: Path | None = None
with left:
    st.markdown('<div class="step-label">步骤 01 · 准备影像</div>', unsafe_allow_html=True)
    st.subheader("选择待预测的 CT")
    if source == "官方示例":
        sample_files = sorted(SAMPLE_DIR.glob("*.nii.gz"))
        if sample_files:
            case_path = sample_files[0]
            st.success(f"已选择官方示例：{case_path.name}")
        else:
            st.error("未找到官方示例数据。")
    else:
        upload = st.file_uploader(
            "上传压缩 NIfTI 文件",
            type=["gz"],
            help="请上传完整的 .nii.gz 文件，无需解压。",
        )
        if upload is not None:
            if not upload.name.endswith(".nii.gz"):
                st.error("文件名必须以 .nii.gz 结尾。")
            else:
                case_path = save_upload(upload)
                st.success(f"已准备病例：{upload.name}")

    if case_path is not None:
        try:
            st.image(preview_slice(str(case_path)), caption="中央轴位切片预览", clamp=True)
        except Exception as exc:
            st.warning(f"影像预览不可用，但仍可尝试推理：{exc}")

with right:
    st.markdown('<div class="step-label">步骤 02 · 运行模型</div>', unsafe_allow_html=True)
    st.subheader("执行 RADAR 推理")
    st.markdown(
        '<div class="research-note">低显存模式覆盖完整 CT，但会使用更多滑窗。'
        '首次运行需要加载模型，预计约 1–3 分钟。</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    run_disabled = not ready or case_path is None
    if st.button("开始预测", type="primary", width="stretch", disabled=run_disabled):
        with st.status("正在运行 RADAR…", expanded=True) as status:
            st.write("加载预训练权重与文本编码器")
            st.write(f"使用滑窗：{roi_size.replace(',', ' × ')}")
            try:
                result_path, log = run_inference(case_path.parent, roi_size)
                st.session_state["result_path"] = str(result_path)
                st.session_state["run_log"] = log
                status.update(label="预测完成", state="complete", expanded=False)
            except subprocess.TimeoutExpired:
                status.update(label="预测超时", state="error")
                st.error("推理超过 60 分钟，已停止等待。")
            except Exception as exc:
                status.update(label="预测失败", state="error")
                st.error("推理未完成。请展开技术日志查看原因。")
                st.code(str(exc), language="text")

result_path_value = st.session_state.get("result_path")
if not result_path_value:
    default_result = RESULTS_DIR / "RADAR_infer_results_demo.csv"
    if default_result.exists():
        result_path_value = str(default_result)

if result_path_value and Path(result_path_value).exists():
    result_path = Path(result_path_value)
    raw_results = pd.read_csv(result_path)
    long_results = to_long_results(raw_results)
    filtered = long_results[long_results["阳性概率"] >= threshold]

    st.divider()
    st.markdown('<div class="step-label">步骤 03 · 查看结果</div>', unsafe_allow_html=True)
    st.subheader("影像征象预测")
    metrics = st.columns(3)
    metrics[0].metric("已输出征象", len(long_results))
    metrics[1].metric(f"概率 ≥ {threshold:.0%}", len(filtered))
    metrics[2].metric("最高概率", f"{long_results['阳性概率'].max():.1%}")

    chart_data = filtered.head(15).sort_values("阳性概率")
    if chart_data.empty:
        st.info("当前阈值下没有可展示的征象，请降低左侧阈值。")
    else:
        st.bar_chart(
            chart_data.set_index("影像征象")["阳性概率"],
            horizontal=True,
            color="#6d28d9",
            height=460,
        )

    st.dataframe(
        long_results,
        width="stretch",
        hide_index=True,
        column_config={
            "阳性概率": st.column_config.ProgressColumn(
                "阳性概率", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    st.download_button(
        "下载原始预测 CSV",
        data=result_path.read_bytes(),
        file_name=result_path.name,
        mime="text/csv",
        width="stretch",
    )

    log_value = st.session_state.get("run_log")
    if log_value:
        with st.expander("技术日志"):
            st.code(log_value, language="text")

