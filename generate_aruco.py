import cv2
import numpy as np

# 1. 選擇我們系統中使用的字典 (4x4_50)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 2. 生成 ID 為 0 的標籤，大小為 400x400 像素
# 注意：這裡的像素大小只是圖片解析度，真實大小還是看你印表機怎麼印
marker_image = cv2.aruco.generateImageMarker(aruco_dict, id=0, sidePixels=400)

# 3. 儲存成圖片檔案
cv2.imwrite("marker_id_0.png", marker_image)
print("✅ 成功生成 marker_id_0.png，請去資料夾查看並列印！")