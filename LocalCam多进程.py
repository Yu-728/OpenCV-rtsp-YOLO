"""
多进程对rstp视频流进行图像处理
现存在问题：笔记本算力不够，cpu和内存爆了
"""

import os
import cv2 as cv
import gc
from multiprocessing import Process, Manager
import torch

# 这里，换成自已的模型，调用yolov5s.pt
model = torch.hub.load('./yolov5', 'custom', path='./weight/yolov5s.pt', source='local')
# 超参数设置
model.conf = 0.52  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)


# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# def img_resize(image):
#     """
#     更改图片尺寸
#     """
#     height, width = image.shape[0], image.shape[1]
#     # 设置新的图片分辨率框架 640x369 1280×720 1920×1080
#     width_new = 1280
#     height_new = 720
#     # 判断图片的长宽比率
#     if width / height >= width_new / height_new:
#         img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
#     else:
#         img_new = cv2.resize(image, (int(width * height_new / height), height_new))
#     return img_new

def yolo_deal(img):
    # Inference
    results = model(img)

    # Results
    # 显示相关结果信息，如：图片信息、识别速度...
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    person = results.xyxy[0]  # 识别结果用tensor保存，包含标签、坐标范围、IOU
    # GPU 转 CPU
    person = person.cpu()
    # tensor 转 array
    person = person.numpy()  # tensor --> array格式
    # 控制台显示
    print(person)
    print(results.xyxy[0])  # img1 predictions (tensor)
    print('----------------')
    print(results.pandas().xyxy[0])  # img1 predictions (pandas)
    # 显示
    # 框出所识别的物体
    cv.rectangle(img, (int(person[0, 0]), int(person[0, 1])), (int(person[0, 2]), int(person[0, 3])),
                 (0, 0, 255), 3)  # 框出识别物体
    return img


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    while True:
        if len(stack) != 0:
            value = stack.pop()
            # 对获取的视频帧分辨率重处理
            # img_new = img_resize(value)
            # 使用yolo模型处理视频帧
            yolo_img = yolo_deal(value)
            # 显示处理后的视频帧
            cv.imshow("img", yolo_img)
            # 将处理的视频帧存放在文件夹里
            # save_img(yolo_img)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break


if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    url = 'rtsp://admin:admin@192.168.43.229:8554/live'
    pw = Process(target=write, args=(q, url, 100))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()

    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
