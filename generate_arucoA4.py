import cv2
import numpy as np

def generate_a4_markers(marker_size_mm=50.0):
    print("🖨️ 正在生成 300 DPI 高精度 A4 標籤排版圖...")

    # --- 1. 定義 A4 紙張與 300 DPI 解析度 ---
    DPI = 300
    px_per_mm = DPI / 25.4  # 每公釐的像素數量 (約 11.81 px/mm)
    
    # A4 紙張真實大小 (210mm x 297mm) 轉成 300 DPI 像素
    a4_width_px = int(210 * px_per_mm)
    a4_height_px = int(297 * px_per_mm)
    
    # 建立一張全白的 A4 畫布
    a4_canvas = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255

    # --- 2. 設定標籤參數 ---
    marker_size_px = int(marker_size_mm * px_per_mm)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 我們打算印 2 欄 3 列，共 6 個標籤 (ID: 0~5)
    cols, rows = 2, 3
    margin_x = (a4_width_px - (cols * marker_size_px)) // 3   # 水平間距
    margin_y = (a4_height_px - (rows * marker_size_px)) // 4  # 垂直間距

    # --- 3. 開始繪製標籤 ---
    marker_id = 0
    for row in range(rows):
        for col in range(cols):
            # 產生標籤影像
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
            
            # 計算貼上的座標位置
            start_x = margin_x + col * (marker_size_px + margin_x)
            start_y = margin_y + row * (marker_size_px + margin_y)
            
            # 將標籤貼到 A4 畫布上
            a4_canvas[start_y:start_y+marker_size_px, start_x:start_x+marker_size_px] = marker_img
            
            # (加分項) 在標籤下方寫上 ID 和真實尺寸，方便你辨認
            text = f"ID: {marker_id} ({int(marker_size_mm)}x{int(marker_size_mm)} mm)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            text_x = start_x + (marker_size_px - text_size[0]) // 2
            text_y = start_y + marker_size_px + text_size[1] + 50
            cv2.putText(a4_canvas, text, (text_x, text_y), font, font_scale, 0, thickness)
            
            marker_id += 1

    # --- 4. 儲存檔案 ---
    filename = "A4_ArUco_Markers_300DPI.png"
    cv2.imwrite(filename, a4_canvas)
    print(f"✅ 成功生成檔案：{filename}")
    print("💡 列印秘訣：請直接使用此圖片列印，並確保印表機設定為【實際大小 / 100% 比例】！")

# 執行腳本 (你可以隨時把 50.0 改成你想要的尺寸)
generate_a4_markers(marker_size_mm=50.0)