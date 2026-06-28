import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_real_object_curve(image_path):
    print(f"🔍 正在讀取真實世界圖片: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 找不到圖片，請檢查檔名是不是 real_test.jpg！")
        return

    # 1. 等比例縮放與 ROI 裁切
    h, w = img.shape[:2]
    scale = 800 / h
    img = cv2.resize(img, (int(w * scale), 800))
    img = img[40:-40, 40:-40]

    # 2. 灰階、高斯模糊與 Canny 邊緣偵測
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # 3. 尋找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    canvas = np.zeros_like(img)

    if len(contours) > 0:
        # 找出最長的主結構 (電線 / 懸鏈線)
        longest_contour = max(contours, key=lambda c: cv2.arcLength(c, closed=False))
        
        cv2.drawContours(canvas, [longest_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(img, [longest_contour], -1, (0, 255, 0), 2)
        
        # ==========================================
        # ✨ 終極融合：將實體輪廓轉化為工程座標數據
        # ==========================================
        print("\n🚀 啟動逆向工程：正在擷取結構變形座標...")
        
        # OpenCV 預設的輪廓陣列形狀是 (N, 1, 2)，我們將其攤平成 (N, 2)
        points = longest_contour.reshape(-1, 2)
        print(f"✅ 成功抓取到 {len(points)} 個物理結構資料點！")
        
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # 先顯示影像處理的結果
        cv2.imshow("1. Real-World Target", img)
        cv2.imshow("2. Extracted Edge", canvas)
        print("👀 請按鍵盤『任意鍵』關閉圖片視窗，系統將自動生成結構分析圖表...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ==========================================
        # 繪製數位化後的數據散佈圖 (力學圖表展示)
        # ==========================================
        plt.figure(figsize=(10, 6))
        
        # 這次我們用顯眼的紅色點雲來標示結構變形
        plt.scatter(x_coords, -y_coords, s=2, c='red')
        
        plt.title("Real-World Structural Deformation Digitization")
        plt.xlabel("X Coordinate (Pixels)")
        plt.ylabel("Y Coordinate (Inverted)")
        plt.grid(True)
        
        # 絕對不能忘記的靈魂：鎖定真實物理比例！
        plt.axis('equal')  
        
        plt.show()

    else:
        print("⚠️ 找不到任何邊緣。")

# 執行程式碼
extract_real_object_curve('real_test.jpg')