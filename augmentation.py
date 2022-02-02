import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)


def read_xml_annotation(root, img_id):
    in_file = open(os.path.join(root, img_id))
    tree = ET.parse(in_file)
    root = tree.getroot()  # 获取根节点
    bbox_list = []

    for obj in root.findall('object'):  # 遍历所有的物体
        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        bbox_list.append([xmin, ymin, xmax, ymax])

    return bbox_list


def change_xml_annotation(root, img_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(img_id) + '.xml'))
    tree = ET.parse(in_file)
    xml_root = tree.getroot()

    object = xml_root.find('object')
    bbox = xml_root.find('bndbox')
    xmin = bbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str("%06d" % (str(id) + '.xml'))))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    # print("当前id：", id)

    in_file = open(os.path.join(root, str(image_id) + '.xml'))
    tree = ET.parse(in_file)
    elem = tree.find('filename')
    elem.text = (str("%06d" % int(id)) + '.jpg')
    xmlroot = tree.getroot()
    index = 0

    for obj in xmlroot.findall('object'):
        bndbox = obj.find('bndbox')

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str("%06d" % int(id)) + '.xml'))


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":
    IMG_DIR = "D:\py_projects/voc_aug/dataset/img"
    XML_DIR = "D:\py_projects/voc_aug/dataset/annotation"

    AUG_IMG_DIR = "./aug_imgs"
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUG_XML_DIR = "./aug_xmls"
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    # 增强因子，每张图片增强的数量
    AUGLOOP = 30

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 数据增强
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # 对50%的图像进行上下翻转
        iaa.Fliplr(0.5),  # 对50%的图像进行左右镜像
        # Multiply 50% of all images with a random value between 0.5 and 1.5 and multiply the remaining 50% channel-wise
        iaa.Multiply((0.5, 1.5), per_channel=0.5),  # 改变亮度，不影响bbox标注
        # Modify the contrast of images according to 255*((v/255)**gamma),
        # where v is a pixel value and gamma is sampled uniformly from the interval [0.5, 2.0] (once per image)
        iaa.GammaContrast((0.5, 2.0)),
        iaa.GaussianBlur(sigma=(0, 3.0)),  # 高斯模糊，不影响bbox标注
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        )  # 位移，影响bbox
    ])

    for root, subfolders, files in os.walk(XML_DIR):

        for name in files:

            # 读取原数据集的坐标, format=[xmin, ymin, xmax, ymax]
            bbox_list = read_xml_annotation(XML_DIR, name)

            # print("原坐标", bbox_list)
            shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)  # 将增强前的xml文件复制到新的路径中
            shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.jpg'), AUG_IMG_DIR)  # name为去除文件后缀的文件名 e.g. img_0174

            for epoch in range(AUGLOOP):
                print("------------------epoch %d-----------------------" % epoch + 1)

                seq_det = seq.to_deterministic()  # 确定一个增强序列，保持坐标和图像同步改变
                # 读取原图像集
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                img = np.asarray(img)
                # bbox坐标增强
                for i in range(len(bbox_list)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bbox_list[i][0], y1=bbox_list[i][1], x2=bbox_list[i][2], y2=bbox_list[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # print("新坐标:", bbs_aug)
                    # 将bbox的坐标限制到[1, img.shape]，以防止训练时报错
                    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    # 防止框左右点重合
                    if n_x1 == 1 and n_x1 == n_x2:
                        n_x2 += 1
                    if n_y1 == 1 and n_y2 == n_y1:
                        n_y2 += 1
                    if n_x1 >= n_x2 or n_y1 >= n_y2:
                        print('error', name)
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])

                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR,
                                    str("%06d" % (len(files) + int(name[4: 8]) + epoch * 250)) + '.jpg')
                image_auged = bbs.draw_on_image(image_aug, size=0)
                Image.fromarray(image_auged).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                           len(files) + int(name[4: 8]) + epoch * 250)
                print(str("%06d" % (len(files) + int(name[4: 8]) + epoch * 250)) + '.jpg')
                new_bndbox_list = []
