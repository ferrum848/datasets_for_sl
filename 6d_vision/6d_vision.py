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


image_class = {'obj0': [0, 0, 0], 'obj1': [20, 20, 20], 'obj2': [40, 40, 40], 'obj3': [80, 80, 80], 'obj4': [160, 160, 160], 'obj5': [255, 255, 255], 'obj6': [100, 100, 100], 'obj7': [140, 140, 140], 'obj8': [60, 60, 60], 'obj9': [240, 240, 240]}
image_regions = {0: 'obj0', 1001001: 'obj1', 2002002: 'obj2', 4004004: 'obj3', 8008008: 'obj4', 255255255: 'obj5', 5005005: 'obj6', 7007007: 'obj7', 3003003: 'obj8', 12012012: 'obj9'}



classes = []
for title, color in image_class.items():
    temp = {'title': title, 'shape': 'bitmap', 'color': color2code(color)}
    classes.append(temp)
meta = {'classes': classes, 'tags_images': [], "tags_objects": []}
json_dump(meta, '/work/datasets/graz/6d/my_project/meta.json')


runner = os.walk('/work/datasets/graz/6d/original_foto/')

sch = 1
for dir, subdir, file in runner:
    if len(file) == 0:
        continue
    for object in os.listdir(dir + '/'):
        name = object[:-4] #imgleft000000009
        print(name) #label000000009
        im = cv2.imread(dir + '/' + object)
        im *= 12
        image = cv2.imread(dir[:23] + 'labels_foto' + dir[36:-7] + 'groundtruth/' + 'label' + name[-9:] + '.pgm')
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.uint32)
        new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
        new_mask = new_mask + (image[:, :, 0] * 1000000) + (image[:, :, 1] * 1000) + image[:, :, 2]

        foto_objects = []
        json_for_image = {'tags': [],
                           'description': '',
                            'objects': foto_objects,
                                  'size': {
                                      'width': image.shape[1],
                                      'height': image.shape[0] }}
        for obj in np.unique(new_mask):
            classTitle = image_regions[obj]
            if len(np.unique(new_mask)) == 1:
                continue
            mask, obj_new = color_to_gray(new_mask, obj)
            left_coner, mask_bool = coords_alternative(mask, obj_new)
            for i in range(len(left_coner)):
                mask_bool[i] = mask_bool[i].astype(np.bool)
                data = mask_2_base64(mask_bool[i])
                temp = {"bitmap":
                            {"origin": [left_coner[i][1], left_coner[i][0]],
                             "data": data},
                        "type": "bitmap",
                        "classTitle": classTitle,
                        "description": "",
                        "tags": [],
                        "points": {"interior": [], "exterior": []}}
                foto_objects.append(temp)
        json_dump(json_for_image, '/work/datasets/graz/6d/my_project/dataset/ann/' + str(sch) + name + '.json')
        #shutil.copy(dir + '/' + object, '/work/datasets/graz/6d/my_project/dataset/img/' + str(sch) + name + '.png')
        cv2.imwrite('/work/datasets/graz/6d/my_project/dataset/img/' + str(sch) + name + '.jpg', im)
        sch += 1