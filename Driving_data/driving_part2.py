import numpy as np
import cv2, random, os, time, json, base64, zlib, shutil
from PIL import Image
import scipy.io
import io

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def color2code(color):
    return '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))

def json_dump(obj, fpath):
    with open(fpath, 'w') as fp:
        json.dump(obj, fp)

def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')



def unique_color(image):
    start = time.time()
    mass = {}
    for index, i in np.ndenumerate(image[:, :, 0]):
        mass[index] = [i]
    for index, i in np.ndenumerate(image[:, :, 1]):
        mass[index].append(i)
    for index, i in np.ndenumerate(image[:, :, 2]):
        mass[index].append(i)
    res = []
    for i in mass.values():
        if not i in res:
                res.append(i)
    print('unique_color  ', time.time() - start)
    return res


def unique_color_new(image):
    start = time.time()
    a = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    result = np.array(a).tolist()
    print('unique_color_new  ', time.time() - start)
    return result


def coords(mask, obj):
    start = time.time()
    coord = []
    for index, i in np.ndenumerate(mask):
        if i == obj:
            coord.append(index)
    min_left = (min(coord, key=lambda x: x[0])[0], min(coord, key=lambda x: x[1])[1])
    max_right = (max(coord, key=lambda x: x[0])[0], max(coord, key=lambda x: x[1])[1])
    res = mask[min_left[0] : max_right[0] + 1, min_left[1] : max_right[1] + 1]
    print('coords  ', time.time() - start)
    return min_left, res


def coords_new(mask, obj):
    ret, thresh = cv2.threshold(mask, obj - 1, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    res = []
    for i in range(len(contours)):
        mass = contours[i]
        mass = np.array(mass).tolist()
        mass = sum(mass, [])
        res.extend(mass)
    min_left_temp = (min(res, key=lambda x: x[0])[0], min(res, key=lambda x: x[1])[1])
    min_left = [min_left_temp[1], min_left_temp[0]]
    max_right_temp = (max(res, key=lambda x: x[0])[0], max(res, key=lambda x: x[1])[1])
    max_right = [max_right_temp[1], max_right_temp[0]]
    result = mask[min_left[0]: max_right[0] + 1, min_left[1]: max_right[1] + 1]
    return min_left, result


def coords_alternative(mask, obj):
    ret, thresh = cv2.threshold(mask, obj - 1, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_min_left, all_results = [], []
    for contur in contours:
        mass = np.array(contur).tolist()
        mass = sum(mass, [])
        min_left_temp = (min(mass, key=lambda x: x[0])[0], min(mass, key=lambda x: x[1])[1])
        min_left = [min_left_temp[1], min_left_temp[0]]
        max_right_temp = (max(mass, key=lambda x: x[0])[0], max(mass, key=lambda x: x[1])[1])
        max_right = [max_right_temp[1], max_right_temp[0]]
        result = mask[min_left[0]: max_right[0] + 1, min_left[1]: max_right[1] + 1]
        all_min_left.append(min_left)
        all_results.append(result)
    return all_min_left, all_results



def color_to_gray(new_mask, obj):
    obj1 = (obj - obj%1000000) // 1000000
    obj2 = (obj%1000000 - obj%1000) // 1000
    obj3 = obj%1000
    mask_bool = np.where(new_mask == obj, new_mask, 0)
    ch1 = np.where(mask_bool != obj, mask_bool, obj1)
    ch2 = np.where(mask_bool != obj, mask_bool, obj2)
    ch3 = np.where(mask_bool != obj, mask_bool, obj3)
    ch1 = np.array(ch1, dtype=np.uint8)
    ch2 = np.array(ch2, dtype=np.uint8)
    ch3 = np.array(ch3, dtype=np.uint8)
    mask = cv2.merge((ch1, ch2, ch3))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask, np.unique(mask)[1]
#===============================================================================
#===============================================================================
#===============================================================================

import csv

FILENAME = "/work/datasets/Driving_Dataset/object-dataset/labels.csv"
slovar ={}
with open(FILENAME, "r", newline="") as file:
    reader = csv.reader(file)
    for row in reader:
        temp = row[0].split(' ')
        if temp[0] not in slovar.keys():
            slovar[temp[0]] = [[int(temp[1]), int(temp[2])], [int(temp[3]), int(temp[4])], temp[6][1:len(temp[6]) - 1]]
        else:
            slovar[temp[0]].extend([[int(temp[1]), int(temp[2])], [int(temp[3]), int(temp[4])], temp[6][1:len(temp[6]) - 1]])

print(slovar)
image_class = {'Car': (255, 0, 0),
               'Truck': (0, 0, 255),
               'Pedestrian': (255, 0, 255),
               'trafficLight': (0, 255, 0),
               'biker': (255, 255, 0)}

#make meta.json
classes = []
for title, color in image_class.items():
    temp = {'title': title, 'shape': 'rectangle', 'color': color2code(color)}
    classes.append(temp)
meta = {'classes': classes, 'tags_images': [], "tags_objects": []}
json_dump(meta, '/work/datasets/Driving_Dataset/my_project/meta.json')


for object in os.listdir('/work/datasets/Driving_Dataset/object-dataset/'):
    name = object[:-4]
    print(name)
    if len(name) < 19:
        continue
    image = cv2.imread('/work/datasets/Driving_Dataset/object-dataset/' + object)

    foto_objects = []
    json_for_image = {'tags': [],
                        'description': '',
                        'objects': foto_objects,
                        'size': {
                        'width': image.shape[1],
                     'height': image.shape[0] }}
    try:
        for i in range(0, len(slovar[object]), 3):
            temp = {"bitmap": None,
                            "type": "rectangle",
                            "classTitle": slovar[object][i + 2],
                            "description": "",
                            "tags": [],
                            "points": {"interior": [], "exterior": [slovar[object][i], slovar[object][i + 1]]}}
            foto_objects.append(temp)
        json_dump(json_for_image, '/work/datasets/Driving_Dataset/my_project/dataset/ann/' + name + '.json')
        shutil.copy('/work/datasets/Driving_Dataset/object-dataset/' + object,
                    '/work/datasets/Driving_Dataset/my_project/dataset/img/' + object)
    except KeyError:
        continue





