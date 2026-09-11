import cv2

print("正在测试环境...")
img = cv2.imread('block.jpg')

if img is None:
    print("❌ 找不到图片！请确认图片名字叫 block.jpg 并且和这个 py 文件在同一个文件夹。")
else:
    print("✅ 环境配置成功！图片读取成功，尺寸为：", img.shape)