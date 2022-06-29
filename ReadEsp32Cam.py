"""
读取ESP32发送的视频流，使用opencv解析视频流格式
"""
import urllib.request
import cv2
import numpy as np

url = 'http://192.168.43.69/cam-hi.jpg'  # 改成自己的ip地址+/cam-hi.jpg

while True:
    imgResp = urllib.request.urlopen(url)
    # print(imgResp)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    # print(imgNp)
    img = cv2.imdecode(imgNp, -1)
    # print(img)
    # all the opencv processing is done here
    cv2.imshow('test', img)
    if ord('q') == cv2.waitKey(10):
        exit(0)
