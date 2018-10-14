image_class = {'fon' : (0, 0, 0),
             'object':(255, 255, 255)
             }


image_regions = {1: 'fon',
             255: 'object'
             }


for object in os.listdir('c:/Games/anaconda/work/super/iCoseg/my_project/dataset/img'):
    name = object[:-4]
    image = cv2.imread('c:/Games/anaconda/work/super/iCoseg/segm/' + name + '.png')

    split = cv2.split(image)
    mask = split[0]
    mask = np.where(mask != 0, mask, 1)
    mask = np.where(mask == 255, mask, 1)

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}

    for obj in np.unique(mask):
        print(obj, name)
        classTitle = image_regions[obj]
        mask_bool = np.where(mask == obj, mask, False)
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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/iCoseg/my_project/dataset/ann/' + name + '.json')




