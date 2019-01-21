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
    if obj == 0:
        new_mask = np.where(new_mask != obj, new_mask, 777)
        new_mask = np.where(new_mask == 777, new_mask, 0)
        new_mask = np.where(new_mask != 777, new_mask, 123123123)
        obj = 123123123
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
    mask = cv2.merge((ch1, ch3, ch2))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask, np.unique(mask)[1]

#===============================================================================
#===============================================================================
#===============================================================================





image_class = {'others': (0, 0, 0), 'rover': (1, 1, 1), 'sky': (17, 17, 17), 'car': (33, 33, 33), 'car_groups': (161, 161, 161), 'motorbicycle': (34, 34, 34), 'motorbicycle_group': (162, 162, 162), 'bicycle': (35, 35, 35), 'bicycle_group': (163, 163, 163), 'person': (36, 36, 36), 'person_group': (164, 164, 164), 'rider': (37, 37, 37), 'rider_group': (165, 165, 165), 'truck': (38, 38, 38), 'truck_group': (166, 166, 166), 'bus': (39, 39, 39), 'bus_group': (167, 167, 167), 'tricycle': (40, 40, 40), 'tricycle_group': (168, 168, 168), 'road': (49, 49, 49), 'siderwalk': (50, 50, 50), 'traffic_cone': (65, 65, 65), 'road_pile': (66, 66, 66), 'fence': (67, 67, 67), 'traffic_light': (81, 81, 81), 'pole': (82, 82, 82), 'traffic_sign': (83, 83, 83), 'wall': (84, 84, 84), 'dustbin': (85, 85, 85), 'billboard': (86, 86, 86), 'building': (97, 97, 97), 'bridge': (98, 98, 98), 'tunnel': (99, 99, 99), 'overpass': (100, 100, 100), 'vegatation': (113, 113, 113), 'unlabeled': (255, 255, 255)}

image_regions = {}
for i, j in image_class.items():
    image_regions[j[0]] = i

#make meta.json
classes = []
for title, color in image_class.items():
    temp = {'title': title, 'shape': 'bitmap', 'color': color2code(color)}
    classes.append(temp)
meta = {'classes': classes, 'tags_images': [], "tags_objects": []}
json_dump(meta, '/work/appoloscape/my_project/meta.json')
#=========================================================


runner = os.walk('/work/appoloscape/img/road04_ins/ColorImage/')

for dir, subdir, file in runner:
    if len(file) == 0:
        continue
    for object in os.listdir(dir + '/'):
        name = object[:-4]
        print(name)
        image = cv2.imread(dir[:33] +'Label' + dir[43:] + '/' + name + '_bin.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #print(np.unique(image))

        foto_objects = []
        json_for_image = {'tags': [],
                           'description': '',
                            'objects': foto_objects,
                                  'size': {
                                      'width': image.shape[1],
                                      'height': image.shape[0] }}

        with open(dir[:33] +'Label' + dir[43:] + '/' + name + '.json') as f:
            data = json.load(f)
        polyg = data['objects']
        list_of_objects, list_of_labels = [], []
        for i in polyg:
            label = i['label']
            for j in i['polygons']:
                list_of_labels.append(label)
                list_of_objects.append(j)

        for obj in np.unique(image):
            if len(np.unique(image)) == 1:
                continue
            elif obj in list_of_labels:
                continue
            classTitle = image_regions[obj]
            mask = np.where(image == obj, image, 0)
            left_coner, mask_bool = coords_new(mask, obj)
            mask_bool = mask_bool.astype(np.bool)
            data = mask_2_base64(mask_bool)
            temp = {"bitmap":
                        {"origin": [left_coner[1], left_coner[0]],
                         "data": data},
                    "type": "bitmap",
                    "classTitle": classTitle,
                    "description": "",
                    "tags": [],
                    "points": {"interior": [], "exterior": []}}
            foto_objects.append(temp)

        for i in range(len(list_of_labels)):
            obj, res = list_of_labels[i], list_of_objects[i]
            classTitle = image_regions[obj]
            mask = np.where(image == obj, image, 0)
            min_left_temp = (min(res, key=lambda x: x[0])[0], min(res, key=lambda x: x[1])[1])
            min_left = [min_left_temp[1], min_left_temp[0]]
            max_right_temp = (max(res, key=lambda x: x[0])[0], max(res, key=lambda x: x[1])[1])
            max_right = [max_right_temp[1], max_right_temp[0]]
            result = mask[min_left[0]: max_right[0] + 1, min_left[1]: max_right[1] + 1]

            left_coner, mask_bool = min_left, result

            mask_bool = mask_bool.astype(np.bool)
            data = mask_2_base64(mask_bool)
            temp = {"bitmap":
                            {"origin": [left_coner[1], left_coner[0]],
                             "data": data},
                        "type": "bitmap",
                        "classTitle": classTitle,
                        "description": "",
                        "tags": [],
                        "points": {"interior": [], "exterior": []}}

            foto_objects.append(temp)
        json_dump(json_for_image, '/work/appoloscape/my_project/dataset/ann/' + name + '.json')
        shutil.copy(dir + '/' + object, '/work/appoloscape/my_project/dataset/img/' + object)
#==========================================================================================================
