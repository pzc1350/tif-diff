import streamlit as st
import numpy as np
from PIL import Image
import rasterio
from io import BytesIO

# 导入之前的预测器
from quick_inference import QuickRemoteSensingPredictor

st.set_page_config(page_title="遥感图像快速解译", layout="wide")

st.title("🛰️ 遥感图像快速解译系统")
st.markdown("上传遥感图像，使用预训练模型进行快速语义分割")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 设置")
    
    model_type = st.selectbox(
        "选择模型",
        ["unet", "deeplabv3", "fpn"],
        help="选择要使用的预训练模型"
    )
    
    num_classes = st.slider(
        "类别数量",
        min_value=2,
        max_value=10,
        value=6,
        help="分割的类别数量"
    )
    
    st.markdown("---")
    st.markdown("### 类别说明")
    st.markdown("""
    - 🔴 建筑物
    - ⬜ 道路
    - 🟢 植被
    - 🔵 水体
    - 🟤 裸土
    - ⬛ 其他
    """)

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.header("📤 输入图像")
    uploaded_file = st.file_uploader(
        "选择遥感图像文件",
        type=['tif', 'tiff', 'jpg', 'png'],
        help="支持GeoTIFF和常见图像格式"
    )
    
    if uploaded_file is not None:
        # 显示上传的图像
        if uploaded_file.type in ['image/jpeg', 'image/png']:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图像", use_column_width=True)
        else:
            st.info("GeoTIFF文件已上传")
            
with col2:
    st.header("📊 预测结果")
    
    if uploaded_file is not None:
        if st.button("🚀 开始预测", type="primary"):
            with st.spinner("正在处理..."):
                # 创建预测器
                predictor = QuickRemoteSensingPredictor(
                    model_type=model_type,
                    num_classes=num_classes
                )
                
                # 根据文件类型处理
                if uploaded_file.type in ['image/jpeg', 'image/png']:
                    # 处理普通图像
                    image = np.array(Image.open(uploaded_file))
                    prediction = predictor._predict_patch(image)
                else:
                    # 处理GeoTIFF
                    # 这里需要实际的处理逻辑
                    st.warning("GeoTIFF处理功能开发中...")
                    
                # 显示结果
                st.success("预测完成！")
                
                # 可视化
                color_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
                colors = predictor.colors
                for label, color in colors.items():
                    if label < num_classes:
                        color_mask[prediction == label] = color
                        
                st.image(color_mask, caption="分割结果", use_column_width=True)
                
                # 下载按钮
                result_img = Image.fromarray(color_mask)
                buf = BytesIO()
                result_img.save(buf, format="PNG")
                st.download_button(
                    label="下载结果",
                    data=buf.getvalue(),
                    file_name="segmentation_result.png",
                    mime="image/png"
                )

# 底部信息
st.markdown("---")
st.markdown("### 📝 使用说明")
st.markdown("""
1. 上传遥感图像（支持GeoTIFF、JPEG、PNG格式）
2. 选择合适的模型和参数
3. 点击"开始预测"按钮
4. 查看和下载分割结果
""")

# 运行方式：streamlit run web_app.py