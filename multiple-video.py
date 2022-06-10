import time
import torch
import cv2 as cv


class MultipleTarget:

    def __init__(self):
        """
        初始化
        """
        # 加载训练模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./weight/yolov5s.pt', source='local')
        # 设置阈值
        self.model.conf = 0.52  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)
        # 加载摄像头
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def draw(self, list_temp, image_temp):
        for temp in list_temp:
            name = temp[6]      # 取出标签名
            temp = temp[:4].astype('int')   # 转成int加快计算
            cv.rectangle(image_temp, (temp[0], temp[1]), (temp[2], temp[3]), (0, 0, 255), 3)  # 框出识别物体
            cv.putText(image_temp, name, (int(temp[0]-10), int(temp[1]-10)), cv.FONT_ITALIC, 1, (0, 255, 0), 2)

    def detect(self):
        """
        目标检测
        """
        while True:
            ret, frame = self.cap.read()
            # 如果正确读取帧，ret为True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv.flip(frame, 1)

            # FPS计算time.start
            start_time = time.time()

            # Inference
            results = self.model(frame)
            pd = results.pandas().xyxy[0]
            # 取出对应标签的list
            person_list = pd[pd['name'] == 'person'].to_numpy()
            bus_list = pd[pd['name'] == 'bus'].to_numpy()
            # 框出物体
            self.draw(person_list, frame)
            self.draw(bus_list, frame)
            # end_time
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            # 控制台显示
            results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
            print(results.xyxy[0])  # img1 predictions (tensor)
            print('----------------')
            print(results.pandas().xyxy[0])  # img1 predictions (pandas)

            # FPS显示
            cv.putText(frame, 'FPS:' + str(int(fps)), (30, 50), cv.FONT_ITALIC, 1, (0, 255, 0), 2)

            cv.imshow('results', frame)
            cv.waitKey(10)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv.destroyAllWindows()


test = MultipleTarget()
test.detect()
