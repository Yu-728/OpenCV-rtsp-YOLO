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

# 这里，换成自已的模型，调用best.pt
model = torch.hub.load('./yolov5', 'custom', path='./weight/yolov5s.pt', source='local')

model.conf = 0.52  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
# 设置识别对象（classes文件的标签顺序）
# (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
# model.classes = None
# model.classes = [0, 15, 16, 17]  # 17表示是马，0是人

# Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# img = 'img_t/bus.jpg'
# opencv加载的图片是BGR的使用pytorch要使用RGB的
img = cv.imread(r'yolov5/data/images/bus.jpg')

# Inference
results = model(img)

# Results
# 显示相关结果信息，如：图片信息、识别速度...
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
person = results.xyxy[0]    # 识别结果用tensor保存，包含标签、坐标范围、IOU
# GPU 转 CPU
person = person.cpu()
# tensor 转 array
person = person.numpy()    # tensor --> array格式
# 控制台显示
print(person)
print(results.xyxy[0])  # img1 predictions (tensor)
print('----------------')
print(results.pandas().xyxy[0])  # img1 predictions (pandas)
# for i in range(len(person)):
#     print(person[i])
# results.save()
# results.crop() # 截取检测的像素后，生成单一图片
# results.pandas()

# 显示
# 框出所识别的物体
cv.rectangle(img, (int(person[0, 0]), int(person[0, 1])), (int(person[0, 2]), int(person[0, 3])),
             (0, 0, 255), 3)  # 框出识别物体
cv.imshow('results', img)
cv.waitKey()
cv.destroyAllWindows()






