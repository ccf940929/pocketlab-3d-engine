import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_blue_curve(image_path):
    print(f"🔍 正在讀取圖片: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 找不到圖片，請檢查檔名！")
        return

    img = cv2.resize(img, (800, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. 顏色過濾
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 顯示處理結果 (把中文標題拿掉，避免亂碼)
    cv2.imshow("1. Original", img)
    cv2.imshow("2. Mask", mask)
    print("👀 請按鍵盤『任意鍵』關閉圖片視窗，讓系統繼續擷取座標...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ==========================================
    # 🚀 競賽核心功能：圖表數位化 (提取像素座標)
    # ==========================================
    print("\n🚀 開始進行數據逆向工程：擷取像素座標...")
    
    # 找出 Mask 中所有「非零 (也就是白色)」的像素座標點
    points = cv2.findNonZero(mask)

    if points is not None:
        # OpenCV 回傳的陣列形狀是 (N, 1, 2)，我們把它壓平變成純粹的 (N, 2) 座標陣列
        points = points.reshape(-1, 2)
        print(f"✅ 成功抓取到 {len(points)} 個像素資料點！")
        print("📝 前 10 個點的座標如下 (X, Y)：\n", points[:10])

        # 把 X 陣列與 Y 陣列拆開
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # ==========================================
        # ✨ CTO 升級魔法：筆畫骨架化 (取平均值)
        # ==========================================
        # 找出所有不重複的 X 座標
        unique_x = np.unique(x_coords)
        # 針對每一個 X，計算其所有 Y 座標的平均值
        averaged_y = np.array([np.mean(y_coords[x_coords == x]) for x in unique_x])

        # 繪製數位化後的數據散佈圖
        plt.figure(figsize=(8, 6))
        
        # 改用瘦身後的 unique_x 和 averaged_y 來畫圖，並加上連線
        plt.plot(unique_x, -averaged_y, marker='o', markersize=2, linestyle='-', color='blue', linewidth=1)
        
        plt.title("Digitized Curve Data (Graph Reverse Engineering)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate (Inverted)")
        plt.grid(True)
        
        # ==========================================
        # ✨ CTO 升級魔法：鎖定真實比例尺
        # ==========================================
        plt.axis('equal')  # 這行是靈魂！強制 X 軸與 Y 軸比例 1:1，絕對不變形！
        
        plt.show()
    else:
        print("⚠️ 找不到任何曲線，請確認 Mask 是否為全黑。")

# ==========================================
# 執行程式碼 (記得確保檔名正確)
extract_blue_curve('your_test_curve.jpg')