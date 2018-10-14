import string
import shelve
import webbrowser, sys, requests, bs4, openpyxl, pprint, PyPDF2, docx, datetime, subprocess
from matplotlib import pyplot as plt
import numpy as np
import cv2, random, os, time, json
from xml.etree.ElementTree import ElementTree
import pandas as pd, json, zlib, base64, io, shutil
from PIL import Image
import scipy.io
import threading


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
    return res


def coords(mask, obj):
    coord = []
    for index, i in np.ndenumerate(mask):
        if i == obj:
            coord.append(index)

    min_left = (min(coord, key=lambda x: x[0])[0], min(coord, key=lambda x: x[1])[1])
    max_right = (max(coord, key=lambda x: x[0])[0], max(coord, key=lambda x: x[1])[1])
    #print(min_left, max_right)
    res = mask[min_left[0] : max_right[0] + 1, min_left[1] : max_right[1] + 1]
    return min_left, res



#================================================================
#main code
#================================================================

image_class = {'water' : (128, 128, 0),
             'grass':(0, 128, 128),
             'rhinoceros or hippo':(128, 0, 0),
             'tree':(128, 0, 128),
             'snow':(0, 0, 128),
             'bear':(0, 128, 0),
             'sky':(128, 128, 128)}


image_regions = {128128000: 'water',
             128128: 'grass',
             128000000: 'rhinoceros or hippo',
             128000128: 'tree',
             128: 'snow',
             128000: 'bear',
             128128128: 'sky'}
'''
#for meta.json
classes = []
for title, color in image_class.items():
    temp = {'title': title, 'shape': 'bitmap', 'color': color2code(color)}
    classes.append(temp)
meta = {'classes': classes, 'tags_images': [], "tags_objects": []}
json_dump(meta, 'c:/Games/anaconda/work/super/Peta/corel/my_project/meta.json')
'''

for object in os.listdir('c:/Games/anaconda/work/super/Peta/corel/my_project/dataset/img'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/Peta/corel/GroundTruth/' + name + '.bmp')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.uint32)

    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    new_mask = new_mask + (image[:, :, 0] * 1000000) + (image[:, :, 1] * 1000) + image[:, :, 2]
    #new_mask = np.where(new_mask != 0, new_mask, 777)

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}

    for obj in np.unique(new_mask):
        classTitle = image_regions[obj]
        mask_bool = np.where(new_mask == obj, new_mask, False)
        #print(obj)
        left_coner, mask_bool = coords(mask_bool, obj)
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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/Peta/corel/my_project/dataset/ann/' + name + '.json')

