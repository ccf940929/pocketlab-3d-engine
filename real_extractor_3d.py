import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
from PIL import Image

def extract_3d_real_curve(image_path):
    print("🔍 1. 正在讀取並優化影像...")
    img = cv2.imread(image_path)
    if img is None: return

    # 保持跟之前一模一樣的尺寸與 ROI 裁切，確保座標系統 100% 對齊
    h, w = img.shape[:2]
    scale = 800 / h
    img = cv2.resize(img, (int(w * scale), 800))
    img = img[40:-40, 40:-40]

    # ---------------------------------------------------------
    # ✨ AI 黑科技：啟動單目深度估計大腦 (Depth Anything)
    # ---------------------------------------------------------
    print("\n🚀 2. 正在載入 AI 深度估計模型 (Depth Anything)...")
    # 使用 Hugging Face pipeline，它會自動下載並執行最適合妳電腦的輕量化模型
    depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    
    # 將 OpenCV 的 BGR 轉換為 AI 看得懂的 PIL RGB 格式
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    print("🔮 AI 正在通靈計算畫面中每個像素的深度距離...")
    ai_result = depth_estimator(pil_img)
    
    # 這是 AI 吐回來的深度圖 (PIL Image)
    depth_pil = ai_result["depth"] 
    # 轉成 Numpy 矩陣：形狀會跟妳裁剪後的 img 尺寸一模一樣！
    depth_map = np.array(depth_pil) 
    
    # ---------------------------------------------------------
    # 傳統 OpenCV：負責精準抓取電線的 (X, Y) 邊緣
    # ---------------------------------------------------------
    print("\n📐 3. 同步啟動 OpenCV 邊緣幾何萃取...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    
    # 強制保留「每一個像素點」(CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) > 0:
        longest_contour = max(contours, key=lambda c: cv2.arcLength(c, closed=False))
        points = longest_contour.reshape(-1, 2)
        print(f"✅ 成功抓取到 {len(points)} 個物理特徵點。")
        
        # ---------------------------------------------------------
        # 🔗 核心交叉引力：虛實數據融合 (X, Y) + Z
        # ---------------------------------------------------------
        print("\n⚡ 4. 正在進行多維數據融合 [(X, Y) 遇見 Z]...")
        
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # 魔法就在這裡：直接拿 OpenCV 的 (x, y) 座標去查 AI 的深度矩陣！
        # 注意：Numpy 矩陣索引是 [row, col]，也就是 [y, x]
        z_depths = np.array([depth_map[y, x] for x, y in points])
        
        # ---------------------------------------------------------
        # 💾 終極落地：自動匯出 3D 結構空間座標 (CSV)
        # ---------------------------------------------------------
        print("\n💾 5. 正在將 3D 點雲數據匯出為 CSV 試算表...")
        
        # 將 X, Y, Z 三個一維陣列，左右合併成一個大表格矩陣
        data_to_export = np.column_stack((x_coords, -y_coords, z_depths))
        
        # 存成 CSV 檔案 (逗號分隔)，並加上資料行的標題
        csv_filename = "structural_deformation_3d.csv"
        np.savetxt(csv_filename, data_to_export, delimiter=",", 
                   header="X_Pixel,Y_Inverted,Z_RelativeDepth", comments='')
                   
        print(f"✅ 儲存成功！高密度測量數據已存入專案資料夾：{csv_filename}")
        
        # ---------------------------------------------------------
        # 📊 成果展示：繪製 3D 空間結構圖 (接續原本畫圖的程式碼...)
        # ---------------------------------------------------------
        # 📊 成果展示：繪製 3D 空間結構圖
        # ---------------------------------------------------------
        fig = plt.figure(figsize=(12, 5))
        
        # 左邊子圖：秀出 AI 算出來的黑白深度圖
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(depth_map, cmap="plasma")
        ax1.set_title("AI Generated Depth Map (Plasma Visual)")
        ax1.axis("off")
        
        # 右邊子圖：畫出高難度的 3D 點雲軌跡圖！
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # 為了符合人類視覺，將 Y 軸和 Z 軸做方向調整
        img_h, img_w = depth_map.shape[:2]
        sc = ax2.scatter(x_coords, -y_coords, z_depths, c=z_depths, cmap="jet", s=3)
        
        ax2.set_title("3D Digitized Structural Curve")
        ax2.set_xlabel("X (Pixels)")
        ax2.set_ylabel("Y (Inverted)")
        ax2.set_zlabel("Z (Relative Depth)")
        fig.colorbar(sc, ax=ax2, label="Proximity (Higher = Closer)")
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("❌ 找不到明顯邊緣。")

# 跑跑看吧！
extract_3d_real_curve('real_test.jpg')