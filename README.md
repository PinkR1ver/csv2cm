# CSV 到 混淆矩阵 (Streamlit 工具)

此工具使用 Streamlit + seaborn + scikit-learn，从 CSV 数据绘制可配置的混淆矩阵。

## 功能
- 读取本地 `data.csv` 或通过页面上传 CSV
- 选择真实标签列与预测列
- 支持多预测列多数投票生成预测
- 可调参数：
  - 标签顺序/子集、归一化方式、注释开关与格式
  - 色图 (cmap)、颜色条开关、色值范围、网格线宽与颜色
  - 图像尺寸、轴标签旋转、字体缩放、下载 PNG（可自定义 DPI）

## 安装
在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

默认会尝试读取同目录下的 `data.csv`；也可以在页面左侧上传新的 CSV。

## CSV 要求
- 第一行应为列名
- 真实/预测列的值将按字符串处理
- 若使用多列投票，空值会被忽略，若出现多众数则取首个

## 目录结构
- `app.py`: Streamlit 前端与绘图逻辑
- `requirements.txt`: 依赖
- `data.csv`: 示例数据（如存在）

## 常见问题
- 注释格式：计数时用 `d`，归一化时建议用 `.2f`
- 标签顺序：若留空则根据数据中的出现顺序自动推断
