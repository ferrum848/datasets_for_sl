'''
classes = []
for title, color in image_class.items():
    temp = {'title': title, 'shape': 'bitmap', 'color': color2code(color)}
    classes.append(temp)
meta = {'classes': classes, 'tags_images': [], "tags_objects": []}
json_dump(meta, 'c:/Games/anaconda/work/super/riemenschneider/meta.json')
'''

image_class = {'void' : (0, 0, 0),
             'building':(128, 0, 0),
             'grass':(0, 128, 0),
             'tree':(128, 128, 0),
             'cow':(0, 0, 128),
             'horse':(128, 0, 128),
             'sheep':(0, 128, 128),
             'sky':(128, 128, 128),
             'mountain':(64, 0, 0),
             'aeroplane':(192, 0, 0),
             'water':(64, 128, 0),
             'face':(192, 128, 0),
             'car':(64, 0, 128),
             'bicycle':(192, 0, 128),
             'flower':(64, 128, 128),
             'sign':(192, 128, 128),
             'bird':(0, 64, 0),
             'book': (128, 64, 0),
             'chair': (0, 192, 0),
             'road': (128, 64, 128),
             'cat': (0, 192, 128),
             'dog': (128, 192, 128),
             'body': (64, 64, 0),
             'boat': (192, 64, 0) }


image_regions = {111: 'void',
             128000000: 'building',
             128000: 'grass',
             128128000: 'tree',
             128: 'cow',
             128000128: 'horse',
             128128: 'sheep',
             128128128: 'sky',
             64000000: 'mountain',
             192000000:'aeroplane',
             64128000:'water',
             192128000:'face',
             64000128:'car',
             192000128:'bicycle',
             64128128:'flower',
             192128128:'sign',
             64000: 'bird',
             128064000:'book',
             192000:'chair',
             128064128:'road',
             192128:'cat',
             128192128:'dog',
             64064000:'body',
             192064000:'boat'}


for object in os.listdir('c:/Games/anaconda/work/super/riemenschneider/my_project/dataset/img'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/riemenschneider/data/GroundTruth/' + name + '_GT.bmp')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.uint32)

    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    new_mask = new_mask + (image[:, :, 0] * 1000000) + (image[:, :, 1] * 1000) + image[:, :, 2]
    new_mask = np.where(new_mask != 0, new_mask, 111)

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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/riemenschneider/my_project/dataset/ann/' + name + '.json')
