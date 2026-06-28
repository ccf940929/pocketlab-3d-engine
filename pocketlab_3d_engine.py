import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
from PIL import Image
from skimage.morphology import skeletonize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
import warnings

def pocketlab_3d_engine_master_v18(image_path, mode='3d', marker_real_size_mm=50.0):
    """
    PocketLab 3D V18 (PCA 轉向 + ArUco 標籤物理抹除防護)
    """
    print(f"🚀 [Step 1] 啟動 PocketLab 3D V18 (場景模式：{mode.upper()})...")
    
    img = cv2.imread(image_path)
    if img is None: 
        print("❌ 找不到圖片！請確認檔名與路徑。")
        return
    
    h, w = img.shape[:2]
    scale = 800 / h
    img = cv2.resize(img, (int(w * scale), 800))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- [Step 2] 掃描環境基準點 (ArUco Markers) ---
    print("🎯 [Step 2] 掃描環境基準點 (ArUco Markers)...")
    mm_per_pixel = None
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(corners) > 0:
        c = corners[0][0]
        pixel_edge_length = np.linalg.norm(c[0] - c[1])
        mm_per_pixel = marker_real_size_mm / pixel_edge_length
        print(f"✅ 發現基準標籤 (ID: {ids[0][0]})！")
        print(f"📏 成功鎖定物理比例尺：1 像素 = {mm_per_pixel:.4f} mm")
        cv2.aruco.drawDetectedMarkers(img_rgb, corners, ids)
    else:
        print("⚠️ 未偵測到基準標籤，系統降級為「相對像素 (Pixel)」模式。")

    # --- [Step 3] 影像前處理雙引擎 ---
    print(f"🧠 [Step 3] 啟動特徵萃取引擎 ({mode.upper()})...")
    if mode == '3d':
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        pil_img = Image.fromarray(img_rgb)
        depth_map = np.array(pipe(pil_img)["depth"]).astype(float)
        depth_8u = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        tophat = cv2.morphologyEx(depth_8u, cv2.MORPH_TOPHAT, kernel_tophat)
        _, binary_mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_to_show = tophat.copy()
    elif mode == '2d':
        binary_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, blockSize=31, C=10
        )
        mask_to_show = binary_mask.copy()
    else:
        return

    # ✨ [V18 核心修復] 物理抹除：將 ArUco 標籤從遮罩中強行挖掉，避免干擾線條判斷
    if ids is not None and len(corners) > 0:
        for c in corners:
            # 在二值化遮罩上，把標籤所在的四邊形塗成全黑 (0)
            cv2.fillPoly(binary_mask, np.int32([c]), 0)

    # 骨架特徵萃取
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        stat_type = cv2.CC_STAT_AREA if mode == '2d' else cv2.CC_STAT_WIDTH
        # 因為標籤被抹掉了，現在面積最大的絕對是妳的鉛筆線！
        largest_label = 1 + np.argmax(stats[1:, stat_type])
        clean_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        clean_mask = binary_mask

    kernel_close = np.ones((5,5), np.uint8)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    skeleton = skeletonize(clean_mask > 0)
    y_coords, x_coords = np.where(skeleton)

    if len(x_coords) == 0: 
        print("❌ 無法萃取出有效曲線。")
        return

    # --- 智慧自適應轉向引擎 (PCA 主成分分析) ---
    orig_x_coords = x_coords.copy()
    orig_y_coords = y_coords.copy()

    coords_matrix = np.vstack((x_coords, y_coords))
    cov_matrix = np.cov(coords_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    primary_vector = eigenvectors[:, np.argmax(eigenvalues)]
    
    is_rotated = False
    if abs(primary_vector[1]) > abs(primary_vector[0]):
        print("🔄 [智慧判定] PCA 偵測到整體趨勢為垂直，自動啟動 90 度數學空間旋轉！")
        x_math = y_coords
        y_math = -x_coords
        is_rotated = True
    else:
        print("➡️ [智慧判定] PCA 偵測到整體趨勢為水平，維持正常座標系。")
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
    print("📐 [Step 4] 執行物理單位轉換與極值分析...")
    unit_str = "Pixel"
    if mm_per_pixel is not None:
        x_local = x_local * mm_per_pixel
        y_local = y_local * mm_per_pixel
        unit_str = "mm"

    if mode == '3d':
        raw_z = depth_map[y_coords[sort_idx], x_coords[sort_idx]]
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

    # --- [Step 5] 渲染儀表板 ---
    print("📊 [Step 5] 渲染全新專業儀表板...")
    fig = plt.figure(figsize=(16, 9)) 
    
    ax1 = fig.add_subplot(2, 2, 1)
    mask_rgb = cv2.cvtColor(mask_to_show, cv2.COLOR_GRAY2RGB)
    img_combined = np.hstack((img_rgb, mask_rgb))
    ax1.imshow(img_combined)
    ax1.plot(img_plot_x, img_plot_y, color='red', linewidth=3) 
    title_suffix = " (PCA Auto-Rotated internally)" if is_rotated else " (Horizontal)"
    ax1.set_title(f"1. Extraction & Marker Detection{title_suffix}")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x_local, y_local, color='black', linewidth=3)
    ax2.set_title(f"2. Extrema Analysis ({unit_str})")
    ax2.set_xlabel(f"X Distance ({unit_str})")
    ax2.set_ylabel(f"Y Deflection ({unit_str})")
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, linestyle='--', alpha=0.6)

    raw_points = [
        {"idx": 0, "name": "Start", "color": "green", "offset": (-20, 15)},
        {"idx": len(x_local)-1, "name": "End", "color": "green", "offset": (20, 15)}
    ]
    for p_idx in peaks: raw_points.append({"idx": p_idx, "name": "Max", "color": "red", "offset": (0, 20)})
    for v_idx in valleys: raw_points.append({"idx": v_idx, "name": "Min", "color": "blue", "offset": (0, -25)})

    final_annotations = []
    dist_threshold = 30 * (mm_per_pixel if mm_per_pixel else 1)
    for p in raw_points:
        idx = p["idx"]
        merged = False
        for existing in final_annotations:
            e_idx = existing["idx"]
            dist = np.hypot(x_local[idx] - x_local[e_idx], y_local[idx] - y_local[e_idx])
            if dist < dist_threshold:
                existing["name"] += f" & {p['name']}"
                if p["name"] in ["Max", "Min"]:
                    existing["color"] = p["color"]
                    existing["offset"] = p["offset"]
                merged = True
                break
        if not merged:
            final_annotations.append(p)

    for ann in final_annotations:
        idx = ann["idx"]
        ax2.scatter(x_local[idx], y_local[idx], color=ann["color"], s=60, zorder=5)
        ax2.annotate(f"{ann['name']}\n({x_local[idx]:.1f}, {y_local[idx]:.1f})",
                     (x_local[idx], y_local[idx]), textcoords="offset points", 
                     xytext=ann["offset"], ha='center', fontsize=10, color=ann["color"], weight='bold')

    if mode == '3d':
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        sc = ax3.scatter(x_local, y_local, z_local, c=z_local, cmap="jet", s=5)
        ax3.set_title(f"3. 3D Spatial View ({unit_str})")
        ax3.set_box_aspect((3, 1, 1))
        fig.colorbar(sc, ax=ax3, shrink=0.5, pad=0.1, label=f"Relative Depth")
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        ax4.text(0.5, 0.5, f"3D Mode Active\nAbsolute Scale: {'YES' if mm_per_pixel else 'NO (Relative)'}", 
                 transform=ax4.transAxes, ha='center', va='center', fontsize=16)
    else:
        ax3 = fig.add_subplot(2, 2, 3)
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
                if r2 >= 0.985: break 

        tck, u = splprep([x_local, y_local], s=len(x_local)*(3 if mm_per_pixel is None else 0.5), k=3)
        u_new = np.linspace(0, 1, 1000)
        x_bspl, y_bspl = splev(u_new, tck)
        num_control_pts = len(tck[1][0])
        #==========================================,
        #[CTO 擴充] 將 B-Spline 控制節點座標印到終端機,
        #==========================================,
        control_x = tck[1][0]
        control_y = tck[1][1]

        print("\n" + "="*40)
        print(f"💎 B-Spline 數位雙生控制節點 (共 {num_control_pts} 點)")
        print(f"單位: {unit_str}")
        print("="*40)
        for i in range(num_control_pts):
            # 格式化輸出，對齊小數點後兩位
            print(f"Node {i+1:02d}: X = {control_x[i]:>8.2f}, Y = {control_y[i]:>8.2f}")
        print("="*40 + "\n")
        # ========================================== 

        ax3.plot(x_local, y_local, 'k.', markersize=2, label='Raw Data', alpha=0.3)
        ax3.plot(x_local, best_y_poly, 'r-', linewidth=2.5, label=f'Auto Polynomial (Deg {best_deg})')
        ax3.plot(x_bspl, y_bspl, 'b--', linewidth=2.5, label='B-Spline Fit')
        
        ax3.set_title(f"3. Geometrical Tracking ({unit_str})")
        ax3.set_xlabel(f"X Distance ({unit_str})")
        ax3.set_ylabel(f"Y Deflection ({unit_str})")
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.6)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off") 
        eq_parts = []
        for i, coef in enumerate(best_coeffs):
            power = best_deg - i
            if power == 0: eq_parts.append(f"{coef:+.2f}")
            elif power == 1: eq_parts.append(f"{coef:+.1e}x")
            else: eq_parts.append(f"{coef:+.1e}x^{power}")
            
        formatted_eq = ""
        for i in range(0, len(eq_parts), 3):
            formatted_eq += " ".join(eq_parts[i:i+3]) + "\n"
        
        rotate_status = "Activated (Vertical PCA Detected)" if is_rotated else "Standby (Horizontal PCA)"
        info_text = (
            f"=== MEASUREMENT STATUS ===\n"
            f"Scale Unit: {unit_str}\n"
            f"Scale Factor: {f'1px = {mm_per_pixel:.3f}mm' if mm_per_pixel else 'Not Detected'}\n"
            f"Auto-Rotate: {rotate_status}\n\n"
            f"=== AUTO POLYNOMIAL REGRESSION ===\n"
            f"Function: y = {formatted_eq.strip()}\n"
            f"Precision (R-sq) = {best_r2:.4f}\n\n"
            f"=== B-SPLINE DIGITAL TWIN ===\n"
            f"Control Matrix: {num_control_pts} Nodes"
        )
        
        ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, ha='left', va='center',
                 fontsize=13, color='#333333', family='monospace')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 將檔名改回妳那張直立的測試照，再跑一次看看！
    pocketlab_3d_engine_master_v18('3.jpg', mode='2d', marker_real_size_mm=50.0)