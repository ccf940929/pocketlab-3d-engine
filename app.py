import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import warnings
from PIL import Image
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev

# 設定 Streamlit 網頁全寬與主題
st.set_page_config(
    page_title="PocketLab 3D Engine - 智慧視覺量測儀表板",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 讓 UI 看起來更有科技感與進階感
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- 核心演算法引擎 (Streamlit 優化版) -----------------
def process_pocketlab_image(uploaded_file, mode='2d', marker_real_size_mm=50.0):
    """
    PocketLab 3D V18 演算法核心 - 專為 Streamlit 網頁渲染優化
    """
    # 讀取上傳的圖片檔案
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("❌ 圖片解碼失敗，請確認檔案格式。")
        return None
        
    h, w = img.shape[:2]
    scale = 800 / h
    img = cv2.resize(img, (int(w * scale), 800))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- [Step 2] 掃描環境基準點 (ArUco Markers) ---
    mm_per_pixel = None
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 繪製偵測畫面的畫布
    detection_img = img_rgb.copy()
    if ids is not None and len(corners) > 0:
        c = corners[0][0]
        pixel_edge_length = np.linalg.norm(c[0] - c[1])
        mm_per_pixel = marker_real_size_mm / pixel_edge_length
        cv2.aruco.drawDetectedMarkers(detection_img, corners, ids)

    # --- [Step 3] 影像前處理雙引擎 ---
    if mode == '3d':
        # 由於 3D 深度模型在伺服器端部署耗能較大，網頁端在此加入提示，並提供簡化的高動態範圍對比作為 3D 模擬
        st.info("💡 網頁端 3D 模式啟動：本機測試時若未安裝 PyTorch 深度模型，系統將自適應使用對比度高度圖進行渲染。")
        # 模擬深度圖 (以亮度作為局部高度基準)
        depth_8u = cv2.equalizeHist(gray)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        tophat = cv2.morphologyEx(depth_8u, cv2.MORPH_TOPHAT, kernel_tophat)
        _, binary_mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_to_show = tophat.copy()
    else:
        binary_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, blockSize=31, C=10
        )
        mask_to_show = binary_mask.copy()

    # ✨ [V18 核心修復] 物理抹除：將 ArUco 標籤從遮罩中強行挖掉，避免干擾線條判斷
    if ids is not None and len(corners) > 0:
        for c in corners:
            cv2.fillPoly(binary_mask, np.int32([c]), 0)

    # 骨架特徵萃取
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        stat_type = cv2.CC_STAT_AREA if mode == '2d' else cv2.CC_STAT_WIDTH
        largest_label = 1 + np.argmax(stats[1:, stat_type])
        clean_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        clean_mask = binary_mask

    kernel_close = np.ones((5,5), np.uint8)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    skeleton = skeletonize(clean_mask > 0)
    y_coords, x_coords = np.where(skeleton)

    if len(x_coords) == 0: 
        return None

    # --- 智慧自適應轉向引擎 (PCA 主成分分析) ---
    orig_x_coords = x_coords.copy()
    orig_y_coords = y_coords.copy()

    coords_matrix = np.vstack((x_coords, y_coords))
    cov_matrix = np.cov(coords_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    primary_vector = eigenvectors[:, np.argmax(eigenvalues)]
    
    is_rotated = False
    if abs(primary_vector[1]) > abs(primary_vector[0]):
        x_math = y_coords
        y_math = -x_coords
        is_rotated = True
    else:
        x_math = x_coords
        y_math = -y_coords

    sort_idx = np.argsort(x_math)
    x_raw = x_math[sort_idx]
    y_raw = y_math[sort_idx]
    
    img_plot_x = orig_x_coords[sort_idx]
    img_plot_y = orig_y_coords[sort_idx]

    x_local = x_raw - x_raw[0]
    y_local = y_raw - y_raw[0]

    # --- [Step 4] 物理單位轉換與訊號分析 ---
    unit_str = "Pixel"
    if mm_per_pixel is not None:
        x_local = x_local * mm_per_pixel
        y_local = y_local * mm_per_pixel
        unit_str = "mm"

    if mode == '3d':
        # 模擬深度轉換
        raw_z = gray[y_coords[sort_idx], x_coords[sort_idx]].astype(float)
        z_ptp = raw_z.max() - raw_z.min()
        if z_ptp == 0: z_ptp = 1
        z_local = (raw_z - raw_z.min()) / z_ptp
        if mm_per_pixel is not None:
            z_local = z_local * (x_local[-1] * 0.5) 
    else:
        z_local = np.zeros_like(x_local, dtype=float)

    prominence_val = 10 * (mm_per_pixel if mm_per_pixel else 1)
    y_smooth = gaussian_filter1d(y_local, sigma=5)
    peaks, _ = find_peaks(y_smooth, prominence=prominence_val) 
    valleys, _ = find_peaks(-y_smooth, prominence=prominence_val)

    # --- [Step 5] 多項式與 B-Spline 數位雙生重構 ---
    best_deg = 3
    best_r2 = 0
    best_coeffs = None
    best_y_poly = None
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for deg in range(3, 16):
            coeffs = np.polyfit(x_local, y_local, deg)
            p = np.poly1d(coeffs)
            y_fit = p(x_local)
            ss_res = np.sum((y_local - y_fit) ** 2)
            ss_tot = np.sum((y_local - np.mean(y_local)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > best_r2:
                best_r2 = r2
                best_coeffs = coeffs
                best_deg = deg
                best_y_poly = y_fit
            if r2 >= 0.985: 
                break 

    tck, u = splprep([x_local, y_local], s=len(x_local)*(3 if mm_per_pixel is None else 0.5), k=3)
    u_new = np.linspace(0, 1, 1000)
    x_bspl, y_bspl = splev(u_new, tck)
    
    control_x = tck[1][0]
    control_y = tck[1][1]
    num_control_pts = len(control_x)

    # 將所有計算結果打包成字典回傳給 Streamlit UI
    results = {
        "detection_img": detection_img,
        "mask_to_show": mask_to_show,
        "img_plot_x": img_plot_x,
        "img_plot_y": img_plot_y,
        "x_local": x_local,
        "y_local": y_local,
        "z_local": z_local,
        "unit_str": unit_str,
        "mm_per_pixel": mm_per_pixel,
        "is_rotated": is_rotated,
        "peaks": peaks,
        "valleys": valleys,
        "best_deg": best_deg,
        "best_r2": best_r2,
        "best_coeffs": best_coeffs,
        "best_y_poly": best_y_poly,
        "x_bspl": x_bspl,
        "y_bspl": y_bspl,
        "control_x": control_x,
        "control_y": control_y,
        "num_control_pts": num_control_pts
    }
    return results

# ----------------- Streamlit 網頁介面佈局 -----------------

st.markdown('<div class="main-title">PocketLab 3D Engine 🔬</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #4B5563; font-size: 1.1rem;">3S競賽專題：手寫座標視覺辨識與自動曲線擬合行動化系統</p>', unsafe_allow_html=True)

# 側邊欄設定面板
st.sidebar.header("🛠️ 系統量測參數設定")
mode = st.sidebar.selectbox("量測分析維度 (Mode)", ["2D 平面幾何模式", "3D 空間感知模式"])
mode_code = '2d' if "2D" in mode else '3d'

marker_size = st.sidebar.number_input(
    "ArUco 實體標籤尺寸 (mm)", 
    min_value=10.0, 
    max_value=200.0, 
    value=50.0, 
    step=1.0,
    help="請輸入您列印在紙上或模型上的實體 ArUco 標籤真實邊長。"
)

# 影像來源選擇（上傳或相機拍照）
st.sidebar.markdown("---")
st.sidebar.subheader("📸 影像輸入來源")
source_option = st.sidebar.radio("選擇輸入方式", ["上傳圖片檔案", "使用手機/電腦相機拍照"])

uploaded_file = None
if source_option == "上傳圖片檔案":
    uploaded_file = st.sidebar.file_uploader("請選擇影像檔案 (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.sidebar.camera_input("請對準量測目標與 ArUco 標籤拍攝")

# ----------------- 數據處理與渲染 -----------------
if uploaded_file is not None:
    with st.spinner("🚀 PocketLab 核心引擎運算中，請稍候..."):
        # 執行量測核心
        res = process_pocketlab_image(uploaded_file, mode=mode_code, marker_real_size_mm=marker_size)
        
    if res is not None:
        st.success("✅ 量測與數位雙生重構完成！")
        
        # 第一排：系統量測狀態卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # ✨ 【修復點】將可能導致 SyntaxError 的巢狀 f-string 反斜線轉義完全分離提取，保證相容性
            m_pix = res["mm_per_pixel"]
            metric_val = f"1px = {m_pix:.3f} mm" if m_pix is not None else "未偵測 (相對像素)"
            st.metric("比例尺狀態", metric_val)
        with col2:
            st.metric("PCA 自適應旋轉", "已啟動 (垂直)" if res["is_rotated"] else "待命 (水平)")
        with col3:
            st.metric("多項式擬合精度 (R²)", f"{res['best_r2']:.4f}")
        with col4:
            st.metric("B-Spline 節點數", f"{res['num_control_pts']} 點")
            
        st.markdown("---")
        
        # 第二排：影像處理與極值分析圖表
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.subheader("1. 影像特徵萃取與標籤物理抹除防護")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            mask_rgb = cv2.cvtColor(res["mask_to_show"], cv2.COLOR_GRAY2RGB)
            img_combined = np.hstack((res["detection_img"], mask_rgb))
            ax1.imshow(img_combined)
            ax1.plot(res["img_plot_x"], res["img_plot_y"], color='red', linewidth=3)
            ax1.axis("off")
            st.pyplot(fig1)
            
        with fig_col2:
            st.subheader("2. 曲線極值分析與邊界條件定位")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(res["x_local"], res["y_local"], color='black', linewidth=3)
            ax2.set_xlabel(f"X 軸距離 ({res['unit_str']})")
            ax2.set_ylabel(f"Y 軸撓度 ({res['unit_str']})")
            ax2.set_aspect('equal', adjustable='datalim')
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # 標註起點與終點
            ax2.scatter(res["x_local"][0], res["y_local"][0], color="green", s=60, zorder=5)
            ax2.annotate(f"Start\n({res['x_local'][0]:.1f}, {res['y_local'][0]:.1f})",
                         (res["x_local"][0], res["y_local"][0]), textcoords="offset points",
                         xytext=(-20, 15), ha='center', color="green", weight='bold')
                         
            ax2.scatter(res["x_local"][-1], res["y_local"][-1], color="green", s=60, zorder=5)
            ax2.annotate(f"End\n({res['x_local'][-1]:.1f}, {res['y_local'][-1]:.1f})",
                         (res["x_local"][-1], res["y_local"][-1]), textcoords="offset points",
                         xytext=(20, 15), ha='center', color="green", weight='bold')
            
            # 標註極大值點
            for p_idx in res["peaks"]:
                ax2.scatter(res["x_local"][p_idx], res["y_local"][p_idx], color="red", s=60, zorder=5)
                ax2.annotate(f"Max\n({res['x_local'][p_idx]:.1f}, {res['y_local'][p_idx]:.1f})",
                             (res["x_local"][p_idx], res["y_local"][p_idx]), textcoords="offset points",
                             xytext=(0, 15), ha='center', color="red", weight='bold')
                             
            # 標註極小值點
            for v_idx in res["valleys"]:
                ax2.scatter(res["x_local"][v_idx], res["y_local"][v_idx], color="blue", s=60, zorder=5)
                ax2.annotate(f"Min\n({res['x_local'][v_idx]:.1f}, {res['y_local'][v_idx]:.1f})",
                             (res["x_local"][v_idx], res["y_local"][v_idx]), textcoords="offset points",
                             xytext=(0, -25), ha='center', color="blue", weight='bold')
            st.pyplot(fig2)

        st.markdown("---")
        
        # 第三排：幾何追蹤與控制節點
        fig_col3, info_col = st.columns([1.2, 0.8])
        
        with fig_col3:
            st.subheader("3. 雙引擎幾何追蹤與連續函數重構")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.plot(res["x_local"], res["y_local"], 'k.', markersize=2, label='原始資料特徵點', alpha=0.3)
            ax3.plot(res["x_local"], res["best_y_poly"], 'r-', linewidth=2.5, label=f'多項式迴歸 (Deg {res["best_deg"]})')
            ax3.plot(res["x_bspl"], res["y_bspl"], 'b--', linewidth=2.5, label='B-Spline 數位雙生曲線')
            ax3.set_xlabel(f"X 軸距離 ({res['unit_str']})")
            ax3.set_ylabel(f"Y 軸撓度 ({res['unit_str']})")
            ax3.legend(loc='upper right')
            ax3.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig3)
            
        with info_col:
            st.subheader("📊 數位雙生迴歸解析數據")
            
            # 多項式迴歸公式美化輸出
            eq_parts = []
            for i, coef in enumerate(res["best_coeffs"]):
                power = res["best_deg"] - i
                if power == 0: eq_parts.append(f"{coef:+.2f}")
                elif power == 1: eq_parts.append(f"{coef:+.1e}x")
                else: eq_parts.append(f"{coef:+.1e}x^{power}")
                
            formatted_eq = ""
            for i in range(0, len(eq_parts), 2):
                formatted_eq += " ".join(eq_parts[i:i+2]) + " \n"
                
            st.info(f"**多項式分析方程 (Auto Polynomial Degree {res['best_deg']}):**\n\n $y = $ {formatted_eq}")
            
            # 展開顯示 B-Spline 的控制節點座標矩陣
            with st.expander(f"💎 B-Spline 控制節點座標矩陣 (共 {res['num_control_pts']} 點)"):
                node_data = {
                    "節點編號": [f"Node {i+1:02d}" for i in range(res["num_control_pts"])],
                    f"X 座標 ({res['unit_str']})": [f"{x:.2f}" for x in res["control_x"]],
                    f"Y 座標 ({res['unit_str']})": [f"{y:.2f}" for y in res["control_y"]]
                }
                st.table(node_data)
    else:
        st.warning("⚠️ 無法從影像中提取出足夠長度的特徵曲線，請調整光線或拍攝距離後重試。")

else:
    # 歡迎畫面提示
    st.info("👋 歡迎使用 PocketLab 3D Engine 量測 App！請於側邊欄選擇「上傳圖片檔案」或「開啟相機拍攝」來開始進行空間幾何分析。")