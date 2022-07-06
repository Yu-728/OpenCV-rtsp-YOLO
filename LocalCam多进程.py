"""
多进程对rstp视频流进行图像处理
现存在问题：笔记本算力不够，cpu爆了，结果能流畅运行一段时间，延时也低
进程一：读取rtsp视频流
视频流保存使用Manager.list
进程二：使用yolo处理视频流

局域网内实现rtsp协议视频推流
获取rtsp视频流，并用yolo对其进行处理。实现目标检测

@author Yuzzz
"""

import os
import cv2 as cv
import gc
from multiprocessing import Process, Manager
import torch
import time


# 向共享缓冲栈中写入数据,rtsp视频流
def write(stack, cam, top: int) -> None:
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())  # write子进程ID
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


def img_resize(image):
    """
    更改图片尺寸
    """
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架 640x369 1280×720 1920×1080
    width_new = 1280
    height_new = 720
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv.resize(image, (int(width * height_new / height), height_new))
    return img_new

# def save_img(yolo_img, pic_number):
#     cv.imwrite(r'E:/Pytorch_learning/SaveImg/%d.jpg' % pic_number, yolo_img)
#     # cv2.imwrite('File_SavePath/%d.bmp' % (i), reImage)  # 保存图片路径
#     pass

def draw(list_temp, image_temp):
    for temp in list_temp:
        name = temp[6]  # 取出label
        temp = temp[:4].astype('int')  # 转成int加快计算
        cv.rectangle(image_temp, (temp[0], temp[1]), (temp[2], temp[3]), (0, 0, 255), 3)  # 框出识别物体
        cv.putText(image_temp, name, (int(temp[0] - 10), int(temp[1] - 10)), cv.FONT_ITALIC, 1, (0, 255, 0), 2)


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())  # read子进程ID
    # 初始化yolo
    model = torch.hub.load('./yolov5', 'custom', path='./weight/yolov5s.pt', source='local')
    # 超参数设置
    model.conf = 0.52  # confidence threshold (0-1)
    model.iou = 0.45  # NMS IoU threshold (0-1)
    while True:
        if len(stack) != 0:
            value = stack.pop()  # 出栈
            # 对获取的视频帧分辨率重处理
            img_new = img_resize(value)
            # 使用yolo模型处理视频帧
            # yolo_img = yolo_deal(img_new)

            # FPS计算time.start
            start_time = time.time()
            # Inference
            results = model(img_new)
            pd = results.pandas().xyxy[0]  # tensor-->pandas的DataFrame
            # 取出对应标签的list
            person_list = pd[pd['name'] == 'person'].to_numpy()
            bus_list = pd[pd['name'] == 'bus'].to_numpy()
            # 框出物体
            draw(person_list, img_new)
            draw(bus_list, img_new)
            # end_time
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            # FPS显示
            cv.putText(img_new, 'FPS:' + str(int(fps)), (30, 50), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
            cv.imshow('results', img_new)

            # 将处理的视频帧存放在文件夹里
            # pic_number = 0  # 图像数量
            # pic_number += 1
            # save_img(img_new, pic_number)
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
