import os
import argparse
import json

from labelme import utils
import numpy as np
import glob
import PIL.Image
import csv

import xml.etree.ElementTree as ET
import glob 

labelme_xml = []
coco_json = "./coco.json"

def csv2xml(csv_file, data_folder='/home/basic/Downloads/mini_competition_dataset'):
    xmls = []
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # print(row) #['image/scene_000028/frame-000017.jpg', 'mask/scene_000028/frame-000017.png']

            # print(type(row)) # list
            scene = row[0].split('/')[1]
            img_id = row[0].split('/')[-1].split('.')[0]
            xml = data_folder + "/" + "label" + "/" + scene + "/" + img_id + ".xml"
            # print(xml)
            # break
            xmls.append(xml)
    return xmls


def get_xml(label_path):
    pattern = label_path + '/**/*.xml'
    labelme_xml = glob.glob(pattern, recursive=True)
    return labelme_xml

def parse_xml(xml_file):
    # one xml
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

def get_image(xml_root, id, scene):
    image = {}

    file_name = xml_root.findall('filename')[0].text

    for imgsize in xml_root.findall('imagesize'):

        height = imgsize.find('nrows').text
        width = imgsize.find('ncols').text

    # print(file_name)
    #
    image["height"] = int(height)
    image["width"] = int(width)
    image["id"] = id
    image["file_name"] = scene + '/' + file_name

    return image

def get_points(xml_root, xml):
    points = {}


    for obj in xml_root.findall('object'):
        point = []
        label = obj.find('name').text # mint, kusan
        # obj_idx = obj.find('id').text
        label2id = {}
        label2id['kinder'] = 0
        label2id['kusan'] = 1
        label2id['doublemint'] = 2

        # if int(obj_idx) > 2:
        #     print(obj_idx, xml)

        for pts in obj.findall('polygon'):
            for pt in pts.findall('pt'):
                x = pt.find('x').text
                y = pt.find('y').text
                # print(x, y)
                x = float(x)
                y = float(y)
                point.append([x, y])
        try:
            obj_idx = label2id[label]
        except KeyError:
            print('invalid label: ', label)
            continue

        if int(obj_idx) > 2:
            print(obj_idx, xml)

        points[int(obj_idx)] = point 

        # points.extend(point)
        # print(points)
    # print(points)
    return points

def getbbox(points):
    polygons = points
    height = 480
    width = 640
    mask = polygons_to_mask([height, width], polygons)
    return mask2box(mask)


def polygons_to_mask(img_shape, polygons):
    #img_shape= [height, width]
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def mask2box(mask):

    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]

    left_top_r = np.min(rows)  # y
    left_top_c = np.min(clos)  # x

    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)

    return [
        left_top_c,
        left_top_r,
        right_bottom_c - left_top_c,
        right_bottom_r - left_top_r,
    ]    

def get_annotation(points, label, img_id, annID):
    annotation = {}
    # print(np.shape(points))
    # print(points) # shape (n_points, 2), [[x1, y1], ..., ]
    contour = np.array(points)
    x = contour[:, 0]
    y = contour[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    annotation["segmentation"] = [list(np.asarray(points).flatten())]
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = img_id

    annotation["bbox"] = list(map(float, getbbox(points)))

    annotation["category_id"] = int(label)  # self.getcatid(label)
    annotation["id"] = annID # +1 every anno.
    return annotation

def get_category(xml_root):
    categories = []

    for obj in xml_root.findall('object'):
        category = {}
        obj_name = obj.find('name').text
        obj_idx = obj.find('id').text
        # print(obj_name)
        category["supercategory"] = obj_name
        category["id"] = int(obj_idx)
        category["name"] = obj_name
        categories.append(category)

    return categories

def data2coco(images, categories, annotations):
    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations
    return data_coco

def save_json(data_coco, save_json_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_json_path)), exist_ok=True)
    json.dump(data_coco, open(save_json_path, "w"), indent=4)
    print("Saved: ", save_json_path)



class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()



    def data_transfer(self):

        
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)

                self.images.append(self.image(data, num))

                for shapes in data["shapes"]:
                    label = shapes["label"].split("_")
                    if label not in self.label:
                        self.label.append(label)

                    points = shapes["points"]

                    # print(points)

                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))

        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        img = None
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label[0]
        category["id"] = len(self.categories)
        category["name"] = label[0]
        return category

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        annotation["segmentation"] = [list(np.asarray(points).flatten())]
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))

        annotation["category_id"] = label[0]  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)


if __name__ == "__main__":
    path_ = '/home/basic/Downloads/mini_competition_dataset'

    val_csv = path_ + "/val.csv"

    path_save_json = path_ + '/val.json'
    
    tmp = csv2xml(val_csv)
    # tmp = get_xml(path_)
    print(len(tmp))
    annID = 0
    annotations = []
    images = []

    for idx, xml in enumerate(tmp):
        xmlroot = parse_xml(xml)
        # print(xml)

        scene = xml.split('/')[-2]
        # print(scene)
        img = get_image(xmlroot, idx, scene)
        # print(img)
        images.append(img)

        categories = get_category(xmlroot)
        # print(categories)


        points_dic = get_points(xmlroot, xml)

        for label, points in points_dic.items():
            annot = get_annotation(points, label, idx, annID)
            annID += 1
            annotations.append(annot) # annot
        # print(annotations)
        # break

    data_coco = data2coco(images, categories, annotations)
    save_json(data_coco, path_save_json)

    # import argparse

    # parser = argparse.ArgumentParser(
    #     description="labelme annotation to coco data json file."
    # )
    # parser.add_argument(
    #     "labelme_images",
    #     help="Directory to labelme images and annotation json files.",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--output", help="Output json file path.", default="trainval.json"
    # )
    # args = parser.parse_args()
    # labelme_json = glob.glob(os.path.join(args.labelme_images, "*.json"))
    # labelme2coco(labelme_json, args.output)
