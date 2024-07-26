import logging
import os.path
import shutil

import numpy as np
import torch
from PIL import Image, ImageOps
from matplotlib import pyplot as plt, patches
from torchvision.io.image import read_image
import cv2
import my_transforms
from Metrics import Metrics
from evaluation import Evaluation
from exp_code import transforms2
from modelcodes.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, \
    SSD300_VGG16_Weights, ssd300_vgg16

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from my_dataset import COCOdataset, VOCDataSet
import json

from visualization import visualization

# 设置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('result.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

device = "cuda"
coco_root = './data/coco2017'
VOC_root = './data/VOCdevkit'
batchsize = 1


def set_model(modeltype='FRCNN', score_throd=0.7):
    """
    使用工厂方法，设置模型以及参数
    :param modeltype:调用的模型
    :param score_throd: 检测阈值
    :return: 权重和模型
    """
    weights = None
    model = None
    if modeltype == 'FRCNN':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_throd)
    elif modeltype == 'SSD':
        weights = SSD300_VGG16_Weights.DEFAULT
        model = ssd300_vgg16(weights=weights, score_thresh=score_throd)
    return weights, model

def cal_APFD(srtd_tp_list):
    degraded_list = srtd_tp_list[::-1]
    instance_num = len(degraded_list)    # 测试用例总数
    fault_num = sum(degraded_list)   # 发现到的错误总数
    # APFD = 1- (TF1 + TF2 + … + TFn)/(测试用例总数*发现到的错误总数) + 1 / (2*测试用例总数), 其中TFi是第i个错误首次被检测到的测试用例的索引
    APFD = 1 + 1 / (2 * instance_num)
    index_total = 0
    for index, x in enumerate(degraded_list):
        index_total += index if x == 1 else 0
    APFD -= index_total / (instance_num * fault_num)
    return APFD



def Inference(modeltype='FRCNN', datatype='COCO', score_throd=0.7):
    """
    模型推理
    :param modeltype: 调用的模型
    :param datatype: 使用的数据集
    :param score_throd: 检测阈值
    :return: instance-level detection result
    """

    def transxywh(bbx):
        bbx[2] = bbx[2] - bbx[0]
        bbx[3] = bbx[3] - bbx[1]
        return bbx

    # configuration setting
    score_throd_str = f"{int(score_throd * 100):03d}"
    datapath = f'./data/{modeltype}_{datatype}{score_throd_str}.json'

    weights, model = set_model(modeltype=modeltype, score_throd=score_throd)

    data_transform = weights.transforms()

    # set valid dataset
    val_dataset = None
    if datatype == 'COCO':
        val_dataset = COCOdataset(coco_root, "val", data_transform)
    elif datatype == 'VOC':
        data_transform = {
            "val": transforms2.Compose([transforms2.ToTensor()])
        }
        val_dataset = VOCDataSet(VOC_root, "2012", data_transform['val'], "val.txt", ex_transforms=None)

    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batchsize,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=0,
                                                     collate_fn=val_dataset.collate_fn)
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            image = list(img.to(device) for img in image)

            outputs = model(image)
            # 读取预测结果中的所有instance，转化成json，并且使instance和image绑定，即添加"image_id"
            for i, prediction in enumerate(outputs):
                cat_ids = prediction["labels"].cpu()
                bboxs = prediction["boxes"].cpu().numpy().tolist()  # 边框
                scores = prediction['scores'].cpu() # 置信度
                dis_scores = prediction['dis_scores'].cpu().numpy().tolist()
                full_score = prediction['full_scores'].cpu().numpy().tolist() #所有类别的置信度
                for j in range(prediction["labels"].shape[0]):
                    content_dic = {
                        "image_id": int(targets[i]["image_id"].numpy()[0]),
                        "category_id": int(cat_ids[j]),
                        "bbox": transxywh(bboxs[j]),
                        "score": float(scores[j]),
                        "full_score": full_score[j],
                        "dis_score": dis_scores[j]
                    }
                    results.append(content_dic)
        # 保存instance-level文件
        json_str = json.dumps(results, indent=4)
        with open(datapath, 'w') as json_file:
            json_file.write(json_str)


def DeepView(datapath='./data/FRCNN_COCO070.json'):
    with open(datapath) as f1:
        pre_list = json.load(f1)
    # 初始化评估模型
    eva = Evaluation(datatype='COCO')
    eva2 = Evaluation(datatype='COCO')
    fault_list, fault_type_num = eva.get_faulttype(pre_list)  # fault_list代表错误列表，fault_type_num代表错误类型的数目
    """
    论文中提到的错误类型形式化定义为Annotation(x) -> Detection(x), 
    分成两种判别：
    1. 分类错误：例cat->dog
    2.定位错误(IOU<0.5)：例cat->LE
    """
    _, _, tp_list = eva2.get_score_iou(pre_list)  # tp_list代表真阳性/假阳性列表，其中0代表真，1代表假
    result = {}
    metrics = Metrics(pre_list=pre_list, dif_list=None, imgpre_list=None)  # 评判分数矩阵
    result['DeepView'] = metrics.deepview('COCO')  # 以DeepView为评判矩阵的分数
    metric_names = [_[0] for _ in result.items()]   # 获取矩阵中每个test方法的名称
    vis = visualization()   # 可视化
    colorlist = ['xkcd:red', 'xkcd:peach', 'xkcd:green', 'xkcd:light purple', 'xkcd:black', 'xkcd:grey']

    deepview_instance = None
    X = []
    Tro_diversity = []
    for i in range(len(result)):
        metric = result[metric_names[i]]
        zipped = zip(metric, fault_list, tp_list, pre_list)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1])) # 对元组列表按 metric 和 fault_list 进行排序
        srt_ = zip(*sort_zipped)
        srtd_metric_list, srtd_fault_list, srtd_tp_list, srtd_pre_list = [list(x) for x in srt_]
        deepview_instance = srtd_pre_list[::-1] # 反转列表，按降序排序

        # rq3 = RQ3engine(slice=[0.1])
        # rq3.map2image(srtd_pre_list[::-1], metric_names[i], modeltype)

        # 计算并绘制多样性曲线，X代表横坐标（时间变化），Y代表累计发现到的错误类型的数量，Tro_diversity表示理论值
        X, Y, Tro_diversity = vis.plt_fault_type_rauc(srtd_fault_list[::-1], fault_type_num)
        plt.plot(X, Y, label=metric_names[i], color=colorlist[i])

        # 计算并打印有效性和多样性指标，rauc_1-500, rauc_2-1000, rauc_3-2000, rauc_5-5000,rauc_all-全部
        x_list, Tro_effective, y_list, rauc_1, rauc_2, rauc_3, rauc_5, rauc_all = eva.RAUC_cls(
            metric_tplist=srtd_tp_list[::-1])

        # 计算APFD分数
        print(f'{metric_names[i]} APFD = {round(cal_APFD(srtd_tp_list), 3)}')
        logger.info(f'{metric_names[i]} APFD = {round(cal_APFD(srtd_tp_list), 3)}')
        # 表示检测到的错误类型数量占理论最大值的RAUC值。
        print(metric_names[i] + ' diversity: RAUC =' + ' ' + str(round(sum(Y) / sum(Tro_diversity) * 100, 2)))
        # plt.plot(x_list, y_list, label=metric_names[i])
        print(metric_names[i] + ' effectiveness: RAUC-n = ' + str(
            [round(rauc_1 * 100, 2), round(rauc_2 * 100, 2), round(rauc_3 * 100, 2), round(rauc_5 * 100, 2),
             round(rauc_all * 100, 2)]))
        logger.info(f'{metric_names[i]} RAUC-500 = {round(rauc_1 * 100, 2)}')

    # 绘制理论多样性曲线
    plt.plot(X, Tro_diversity, label='Theoretical', color='tab:blue')

    plt.xlabel('$Number\ of\ prioritized\ test\ instances$')
    plt.ylabel('$Number\ of\ error\ type\ detected$')

    plt.legend()
    plt.show()

    return deepview_instance

def transxyxy(bbx):
        return [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]

def draw_gt_boxes(image_path, output_gt_image_file, image_id, id2gtbbx, id2gtlabel):
    """
    绘画真实边框
    :param image_path: 图片路径
    :param output_gt_image_file:图片输出路径
    :param image_id: 图片id
    :param id2gtbbx: id-gtbox字典
    :param id2gtlabel: id-label字典
    """
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    # 将图片转化成RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取真实边框和label
    gt_boxes = id2gtbbx.get(image_id, [])
    gt_labels = id2gtlabel.get(image_id, [])

    # 创建图片
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.axis('off')  # 隐藏坐标轴

    # 绘制真实边框
    for bbox, label in zip(gt_boxes, gt_labels):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='green',
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 10, str(label), color='green', fontsize=12)


    plt.savefig(output_gt_image_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def process_image(deepvirw_result_path, image_path, output_folder, topk = 5, datatype = 'COCO'):
    if datatype == 'COCO':
        with open('./data/instances_val2017.json') as f:
            true_list = json.load(f)
        GT_list = true_list["annotations"]
        id2gtbbx = {x['id']: [] for x in true_list['images']}
        id2gtlabel = {x['id']: [] for x in true_list['images']}
        for x in GT_list:
            id2gtbbx[x['image_id']] = id2gtbbx[x['image_id']] + [transxyxy(x['bbox'])]
            id2gtlabel[x['image_id']] = id2gtlabel[x['image_id']] + [x['category_id']]

    # 读取deepvirw_result.json文件
    assert os.path.exists(deepvirw_result_path)
    with open(deepvirw_result_path) as f:
        deepvirw_result_json = json.load(f)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_image_list = deepvirw_result_json[:topk]
    for image in process_image_list:
        image_id = image['image_id']
        image_id_path = f"{int(image['image_id']):012d}"
        bbox = image['bbox']
        pre_category_id = image['category_id']
        image_file = os.path.join(image_path, f"{image_id_path}.jpg")
        if not os.path.exists(image_file):
            print(f"图像文件 {image_file} 不存在")
            return

        output_gt_image_file = os.path.join(output_folder, f"{image_id_path}_origin.png")
        draw_gt_boxes(image_file, output_gt_image_file, image_id, id2gtbbx, id2gtlabel)

        # 复制图像文件到输出文件夹
        output_image_file = os.path.join(output_folder, f"{image_id_path}.jpg")
        shutil.copy(image_file, output_image_file)
        # 打开图像
        image = Image.open(output_image_file).convert("RGB")

        # 将图像转换为灰度
        gray_image = ImageOps.grayscale(image)
        gray_image = gray_image.convert("RGB")  # 转换为三通道灰度图像

        # 转换为OpenCV图像格式
        image_np = np.array(image)
        gray_image_np = np.array(gray_image)

        # 获取边界框坐标
        x_min, y_min, width, height = map(int, bbox)
        x_max, y_max = x_min + width, y_min + height

        # 将bbox区域的彩色部分覆盖到灰度图像中
        gray_image_np[y_min:y_max, x_min:x_max] = image_np[y_min:y_max, x_min:x_max]
        # 创建一个Matplotlib图像
        fig, ax = plt.subplots(1)
        ax.imshow(gray_image_np)
        ax.axis('off')  # Hide the axis

        # 添加红色边框
        red_color = (1, 0, 0)  # Matplotlib uses RGB
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='red',
                                 facecolor='none')
        ax.add_patch(rect)

        # 在边界框上添加pre_category_id
        ax.text(x_min, y_min - 10, str(pre_category_id), color=red_color, fontsize=12)

        # 保存处理后的图像
        plt.savefig(output_image_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"图像 {image_id_path} 已处理并保存到 {output_image_file},pre_category_id = {pre_category_id}")





if __name__ == "__main__":
    visual_flag = True

    modeltype_list = ['FRCNN', 'SSD']
    datatype_list = ['COCO']
    score_throd_list = {'FRCNN': [0.7, 0.8, 0.9], 'SSD': [0.5, 0.6, 0.7]}
    for modeltype in modeltype_list:
        for datatype in datatype_list:
            for score_throd in score_throd_list[modeltype]:
                logger.info(f'modeltype={modeltype},datatype={datatype},score_throd={score_throd}')
                score_throd_str = f"{int(score_throd * 100):03d}"
                datapath = f'./data/{modeltype}_{datatype}{score_throd_str}.json'

                if not os.path.exists(datapath):
                    Inference(modeltype=modeltype, datatype=datatype, score_throd=score_throd)  # Model Inference

                deepview_result = DeepView(datapath)
                json_str = json.dumps(deepview_result, indent=4)
                with open('./data/deepview_result', 'w') as json_file:
                    json_file.write(json_str)

    if visual_flag:
        process_image('./data/deepview_result', './data/coco2017/val2017/', './data/output_pictures', 100)
