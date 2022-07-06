# 订阅
import binascii
import os
import random
import cv2
import numpy as np
import torch
from paho.mqtt import client as mqtt_client

# --------------MQTT初始化-------------------#
broker = 'broker.emqx.io'
port = 1883
topic = "esp32/test1"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'

# ---------------YOLO初始化------------------#
# 加载训练模型
model = torch.hub.load('./yolov5', 'custom', path='./weight/yolov5s.pt', source='local')
# 设置阈值
model.conf = 0.52  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)


class GlobalValue:
    data = [""]
    temp = 0


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        GlobalValue.data.append(str(msg.payload.decode()))  # 获取十六进制图片数据

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    imgcount = 0
    while True:
        client.loop_start()
        # 判断一张图片是否发完
        if (GlobalValue.data[len(GlobalValue.data) - 1]) == "1":
            # print("Reached EOF")
            data1 = "".join(GlobalValue.data[0:(len(GlobalValue.data) - 1)])
            GlobalValue.data = [""]
            data1 = binascii.a2b_hex(data1)
            data1 = np.frombuffer(data1, dtype=np.uint8)
            img = cv2.imdecode(data1, 1)
            # 保存图片(经测试可使用)
            label = "SaveImg/image" + str(imgcount) + ".jpg"
            with open(label, "wb") as image_file:
                image_file.write(data1)
            cv2.imshow("Door", img)
            # 使用YOLO处理图片
            result = yolo_detect(img)
            cv2.imshow('result', result)
            # 保存处理后的图片 
            label = "ResultImg/image" + str(imgcount) + ".jpg"
            cv2.imwrite(label, result)
            cv2.waitKey(1)
            if imgcount >= 40:
                exit(0)
            else:
                imgcount += 1


def yolo_detect(img):
    # Inference
    results = model(img)
    # Results
    # 显示相关结果信息，如：图片信息、识别速度...
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    person = results.xyxy[0]  # 识别结果用tensor保存数据，包含标签、坐标范围、IOU
    # GPU 转 CPU
    person = person.cpu()
    # tensor 转 array
    person = person.numpy()  # tensor --> array格式
    # 控制台显示
    # print(person)
    # print(results.xyxy[0])  # img1 predictions (tensor)
    # print('----------------')
    # print(results.pandas().xyxy[0])  # img1 predictions (pandas)
    # 显示
    # 框出所识别的物体
    if len(person) > 0:
        cv2.rectangle(img, (int(person[0, 0]), int(person[0, 1])), (int(person[0, 2]), int(person[0, 3])),
                      (0, 0, 255), 3)  # 框出识别物体
    return img


if __name__ == '__main__':
    run()
