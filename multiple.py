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
        # # 加载摄像头
        # self.cap = cv.VideoCapture(0)

    def draw(self, list1, image_temp):

        for temp in list1:
            name = temp[6]      # 取出标签名
            temp = temp[:4].astype('int')   # 转成int加快计算
            # print(temp)
            # print('***************')
            # cv.rectangle(image_temp, (int(temp[0]), int(temp[1])), (int(temp[2]), int(temp[3])),
            #              (0, 0, 255), 3)  # 框出识别物体
            cv.rectangle(image_temp, (temp[0], temp[1]), (temp[2], temp[3]), (0, 0, 255), 3)  # 框出识别物体
            cv.putText(image_temp, name, (int(temp[0]-10), int(temp[1]-10)), cv.FONT_ITALIC, 1, (0, 255, 0), 2)

    def detect(self, image_temp):
        """
        目标检测
        """
        img = image_temp
        # Inference
        results = self.model(img)
        # Results
        # 显示相关结果信息，如：图片信息、识别速度...
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
        pd1 = results.xyxy[0]  # 识别结果用tensor保存，包含标签、坐标范围、IOU
        pd = results.pandas().xyxy[0]   # tensor 转为 DataFrame(Pandas中的数据类型)

        person_list = pd[pd['name'] == 'person'].to_numpy()
        bus_list = pd[pd['name'] == 'bus'].to_numpy()

        self.draw(person_list, img)
        self.draw(bus_list, img)

        # 控制台显示
        # print(pd1)
        # print(pd)
        print(person_list)
        print(results.xyxy[0])
        print('----------------------------------------------------------')
        print(results.pandas().xyxy[0])
        # 窗口显示s
        cv.imshow('results', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


image = cv.imread(r'yolov5/data/images/bus.jpg')
test = MultipleTarget()
test.detect(image)
