import os
import io
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix
import sys


APP_TITLE = "CSV → Confusion Matrix"
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def ensure_fonts_dir() -> None:
	try:
		os.makedirs(FONTS_DIR, exist_ok=True)
	except Exception:
		pass


def register_fonts_from_dir(directory: str) -> List[str]:
	"""注册目录下的字体文件，返回注册到的字体族名称列表。"""
	registered_names: List[str] = []
	if not os.path.isdir(directory):
		return registered_names
	for fname in os.listdir(directory):
		if not fname.lower().endswith((".ttf", ".otf", ".ttc")):
			continue
		path = os.path.join(directory, fname)
		try:
			font_manager.fontManager.addfont(path)
			fp = font_manager.FontProperties(fname=path)
			name = fp.get_name()
			if name and name not in registered_names:
				registered_names.append(name)
		except Exception:
			continue
	return registered_names


def configure_chinese_font(selected_font_name: Optional[str], local_font_names: List[str]) -> Optional[str]:
	"""配置中文字体并返回首选字体名（若可用）。

	- 在 macOS 上优先使用系统内置 PingFang 字体文件，确保中文可用；
	- 其他平台提供回退链；
	- 返回的字符串可用于 fontname 参数强制应用到轴标题与刻度。
	"""
	preferred_name: Optional[str] = None
	# 仅使用本地 fonts/ 中的字体，不再使用系统字体
	available_names = set(local_font_names)
	if selected_font_name and selected_font_name in available_names:
		preferred_name = selected_font_name
	else:
		# 自动选择：按常见中文字体优先级从本地字体中挑选
		priority = [
			"Noto Sans CJK SC",
			"Source Han Sans SC",
			"PingFang SC",
			"Hiragino Sans GB",
			"Microsoft YaHei",
			"SimHei",
			"SimSun",
			"Songti SC",
			"Heiti SC",
			"KaiTi",
		]
		for name in priority:
			if name in available_names:
				preferred_name = name
				break
		if not preferred_name and local_font_names:
			preferred_name = local_font_names[0]

	plt.rcParams["font.family"] = ["sans-serif"]
	if preferred_name:
		# 仅使用本地字体链
		fallback_chain = [preferred_name] + [n for n in local_font_names if n != preferred_name]
		plt.rcParams["font.sans-serif"] = fallback_chain + ["DejaVu Sans"]
	else:
		# 没有本地字体时，退回 DejaVu Sans（无中文亦可运行）
		plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
	plt.rcParams["axes.unicode_minus"] = False
	return preferred_name


def load_csv(upload: Optional[io.BytesIO]) -> pd.DataFrame:
	if upload is not None:
		upload.seek(0)
		return pd.read_csv(upload)
	# fallback removed: must upload
	st.warning("请在左侧上传 CSV 文件以继续")
	st.stop()


def get_label_universe(df: pd.DataFrame, columns: List[str]) -> List[str]:
	labels: List[str] = []
	for col in columns:
		series = df[col].dropna().astype(str)
		labels.extend(series.unique().tolist())
	# stable unique order by first appearance
	seen = set()
	ordered: List[str] = []
	for v in labels:
		if v not in seen:
			seen.add(v)
			ordered.append(v)
	return ordered


def compute_preds_by_majority(df: pd.DataFrame, pred_cols: List[str]) -> pd.Series:
	# majority vote across selected prediction columns (string categories)
	votes = df[pred_cols].astype(str)
	mode_vals = votes.mode(axis=1, dropna=True)
	# If multiple modes, take the first (stable)
	return mode_vals.iloc[:, 0]


def main() -> None:
	st.set_page_config(page_title=APP_TITLE, layout="wide")
	# 字体：准备字体目录并注册本地字体，随后提供上传/选择
	ensure_fonts_dir()
	local_fonts = register_fonts_from_dir(FONTS_DIR)
	st.title(APP_TITLE)
	st.caption("上传 CSV，选择标签列与预测列，绘制可调混淆矩阵")

	with st.sidebar:
		st.header("数据与列选择")
		uploaded = st.file_uploader("上传 CSV 文件", type=["csv"], accept_multiple_files=False)
		try:
			df = load_csv(uploaded)
		except Exception as e:
			st.error(f"读取 CSV 失败: {e}")
			st.stop()

		st.success(f"数据已载入，形状: {df.shape[0]} 行 × {df.shape[1]} 列")
		all_cols = df.columns.tolist()

		true_col = st.selectbox("真实标签列 (y_true)", options=all_cols, index=0 if all_cols else None)

		st.markdown("**预测列设置**")
		use_majority = st.checkbox("多列投票为预测 (多数表决)", value=False)
		if use_majority:
			pred_cols = st.multiselect("选择多个预测列用于投票", options=[c for c in all_cols if c != true_col])
			pred_col_selected = None
		else:
			pred_col_selected = st.selectbox("单列预测 (y_pred)", options=[c for c in all_cols if c != true_col])
			pred_cols = []

		st.divider()
		st.header("字体")
		font_upload = st.file_uploader("上传字体文件 (ttf/otf/ttc)", type=["ttf", "otf", "ttc"], accept_multiple_files=True)
		if font_upload:
			for up in font_upload:
				try:
					ensure_fonts_dir()
					out_path = os.path.join(FONTS_DIR, up.name)
					with open(out_path, "wb") as f:
						f.write(up.getbuffer())
				except Exception as e:
					st.warning(f"字体保存失败: {up.name}: {e}")
			# 重新注册刚上传的字体
			local_fonts = register_fonts_from_dir(FONTS_DIR)
		# 仅列出本地 fonts/ 中注册的字体
		avail = sorted(local_fonts)
		if not avail:
			st.info("未检测到本地字体，请上传 .ttf/.otf/.ttc 到 fonts/ 或通过此处上传。")
		font_choice = st.selectbox("选择字体 (优先使用)", options=["自动"] + avail, index=0)
		selected_font = None if font_choice == "自动" else font_choice
		st.header("矩阵与图形设置")
		candidate_labels = get_label_universe(df, [true_col] + (pred_cols if use_majority else [pred_col_selected]))
		label_order = st.multiselect(
			"标签顺序/子集 (留空=自动)", options=candidate_labels, default=candidate_labels,
			help="可重新排序或筛选，只包含需要展示的标签"
		)

		normalize = st.selectbox(
			"归一化方式",
			options=["none", "true", "pred", "all"],
			index=0,
			help="none=计数；true=按真实类别行归一化；pred=按预测列归一化；all=总体归一化",
		)

		annot = st.checkbox("显示数值注释", value=True)
		fmt = st.text_input("注释格式 (e.g. d, .2f)", value="d" if normalize == "none" else ".2f")
		cmap = st.selectbox("色图 (cmap)", options=sorted(m for m in plt.colormaps() if not m.endswith("_r")), index=sorted(m for m in plt.colormaps() if not m.endswith("_r")).index("Blues") if "Blues" in plt.colormaps() else 0)
		cbar = st.checkbox("显示颜色条", value=True)
		vmin = st.number_input("色值最小 (留空用None)", value=0.0 if normalize != "none" else 0.0)
		vmax = st.number_input("色值最大 (留空用None)", value=1.0 if normalize != "none" else float(df.shape[0]))
		linewidths = st.number_input("网格线宽", value=0.5, min_value=0.0, step=0.1)
		linecolor = st.color_picker("网格线颜色", value="#FFFFFF")
		fig_w = st.number_input("图宽 (英寸)", value=5.0, min_value=2.0, step=0.5)
		fig_h = st.number_input("图高 (英寸)", value=5.0, min_value=2.0, step=0.5)
		xt_rot = st.number_input("X 轴标签旋转", value=45, min_value=0, max_value=90, step=5)
		yt_rot = st.number_input("Y 轴标签旋转", value=0, min_value=0, max_value=90, step=5)
		font_scale = st.slider("字体缩放", min_value=0.6, max_value=2.0, value=1.0, step=0.1)
		thresh = st.slider("注释阈值 (用于反色)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

		# 轴方向（明确 y_true/y_pred 对应的轴）
		axis_orientation = st.selectbox(
			"轴方向",
			options=["行=真实 (y_true), 列=预测 (y_pred) — 标准", "行=预测 (y_pred), 列=真实 (y_true) — 转置"],
			index=0,
		)

		# 轴标题自定义
		default_x_title = "预测标签" if axis_orientation.startswith("行=真实") else "真实标签"
		default_y_title = "真实标签" if axis_orientation.startswith("行=真实") else "预测标签"
		x_axis_title = st.text_input("X 轴标题", value=default_x_title)
		y_axis_title = st.text_input("Y 轴标题", value=default_y_title)
		plot_title = st.text_input("图标题", value="混淆矩阵")

		st.divider()
		st.header("下载与导出")
		img_dpi = st.number_input("导出 DPI", value=200, min_value=72, step=10)

	# 应用字体（用户选择优先）
	font_name = configure_chinese_font(selected_font, local_fonts)

	# Prepare labels
	y_true = df[true_col].astype(str)
	if use_majority and pred_cols:
		y_pred = compute_preds_by_majority(df, pred_cols).astype(str)
	else:
		y_pred = df[pred_col_selected].astype(str)

	labels = label_order if len(label_order) > 0 else get_label_universe(df, [true_col] + (pred_cols if use_majority else [pred_col_selected]))

	norm_param: Optional[str]
	if normalize == "none":
		norm_param = None
	elif normalize == "true":
		norm_param = "true"
	elif normalize == "pred":
		norm_param = "pred"
	else:
		norm_param = "all"

	cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm_param)
	cm_plot = cm if axis_orientation.startswith("行=真实") else cm.T

	sns.set_theme(style="white", font_scale=font_scale)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))
	sns.heatmap(
		cm_plot,
		annot=annot,
		fmt=fmt,
		cmap=cmap,
		cbar=cbar,
		vmin=None if vmin is None else vmin,
		vmax=None if vmax is None else vmax,
		linewidths=linewidths,
		linecolor=linecolor,
		square=True,
		ax=ax,
		annot_kws={"color": "black"},
	)
	if font_name:
		ax.set_xlabel(x_axis_title, fontname=font_name)
		ax.set_ylabel(y_axis_title, fontname=font_name)
		ax.set_title(plot_title, fontname=font_name)
	else:
		ax.set_xlabel(x_axis_title)
		ax.set_ylabel(y_axis_title)
		ax.set_title(plot_title)
	ax.set_xticklabels(labels, rotation=xt_rot, ha="right")
	ax.set_yticklabels(labels, rotation=yt_rot)
	# 强制刻度文字使用中文字体（如可用）
	if font_name:
		for t in ax.get_xticklabels() + ax.get_yticklabels():
			t.set_fontname(font_name)
	col1, col2 = st.columns([1, 1])
	with col1:
		st.pyplot(fig, clear_figure=True)
		buf = io.BytesIO()
		fig.savefig(buf, format="png", dpi=img_dpi, bbox_inches="tight")
		buf.seek(0)
		st.download_button("下载图像 (PNG)", data=buf, file_name="confusion_matrix.png", mime="image/png")
	with col2:
		with st.expander("查看数据预览"):
			st.dataframe(df.head(50))


if __name__ == "__main__":
	main()
