import time
import torch
import cv2 as cv

# Model
"""
def load(repo_or_dir, model, *args, source='github', force_reload=False, verbose=True, skip_validation=False,
         **kwargs):
从 github 存储库或本地目录加载模型。

注意：加载模型是典型的用例，但这也可用于加载其他对象，例如分词器、损失函数等。

如果source是“github”，repo_or_dir则应repo_owner/repo_name[:tag_name]采用带有可选标签/分支的形式。

如果source是“local”，repo_or_dir则应为本地目录的路径。

参数

repo_or_dir ( string ) – 如果source是 'github'，这应该对应于repo_owner/repo_name[:tag_name]具有可选标签/分支格式的 github 存储库，例如 'pytorch/vision:0.10'。如果tag_name未指定，则假定默认分支为main存在，否则为master。如果source是“local”，则它应该是本地目录的路径。

model ( string ) – 在 repo/dir's 中定义的可调用（入口点）的名称hubconf.py。

*args（可选）– callable 的相应参数model。

source ( string , optional ) – 'github' 或 'local'。指定如何 repo_or_dir解释。默认为“github”。

force_reload ( bool , optional ) – 是否无条件强制重新下载github repo。如果 没有任何影响 source = 'local'。默认为False

verbose ( bool , optional ) – 如果False，静音有关命中本地缓存的消息。请注意，有关首次下载的消息无法静音。如果source = 'local'没有任何影响。默认为True。

skip_validation ( bool , optional ) – 如果False，torchhub 将检查github参数指定的分支或提交是否正确属于 repo 所有者。这将向 GitHub API 发出请求；您可以通过设置GITHUB_TOKEN环境变量来指定非默认 GitHub 令牌 。默认为False。

**kwargs (可选) – callable 的相应 kwargs model。

返回

model使用给定*args和调用时可调用 的输出**kwargs。

"""
# yolov5s表示主目录下的yolov5s.pt，而且必须是主目录下
# 网络模型从github上加载      yolov5路径  模型名
# model = torch.hub.load('./yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom-->表示使用自己的训练模型

# 这里，换成自已的模型
model = torch.hub.load('./yolov5', 'custom', path='./weight/ppe_yolo_n.pt', source='local')

model.conf = 0.52  # confidence threshold (0-1)
model.iou = 0.48  # NMS IoU threshold (0-1)
# 设置识别对象（classes文件的标签顺序）
# (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
# model.classes = None
label = ['person', 'vest', 'blue helmet', 'red helmet', 'white helmet', 'yellow helmet']

# Images
# 图片输入
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# img = 'img_t/bus.jpg'
# opencv加载的图片是BGR的使用pytorch要使用RGB的
# img = cv.imread(r'image/12.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 从摄像头获取
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获，这里获取到得图像是镜像的，后续需要对其进行处理
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 1)   # frame为BGR
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # FPS计算time.start
    start_time = time.time()

    # Inference
    results = model(img)
    person = results.xyxy[0]  # xyxy[0]--> 0在classes是person标签,获取所有标签为0的数据
    # GPU --> CPU
    person = person.cpu()
    # tensor --> array
    person = person.numpy()

    # end_time
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    print(results.xyxy[0])  # img1 predictions (tensor)
    print('----------------')
    print(results.pandas().xyxy[0])  # img1 predictions (pandas)
    # results.save()
    # results.crop() # 截取检测的像素后，生成单一图片
    # results.pandas()

    # 显示
    cv.putText(frame, 'FPS: ' + str(round(fps, 2)), (30, 50), cv.FONT_ITALIC, 1, (255, 255, 0), 2)
    if len(person) > 0:
        cv.putText(frame, 'Person: ' + str(int(person[0, 4] * 100)) + '%', (30, 100), cv.FONT_ITALIC, 1, (0, 255, 0), 2)
        cv.rectangle(frame, (int(person[0, 0]), int(person[0, 1])), (int(person[0, 2]), int(person[0, 3])),
                     (0, 0, 255), 3)  # 人框

    cv.imshow('results', frame)
    cv.waitKey(10)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
