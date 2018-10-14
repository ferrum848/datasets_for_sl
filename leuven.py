image_class = {'building' : (128, 0, 0),
             'unknown':(0, 0, 0),
             'sky':(128, 128, 128),
             'car':(64, 0, 128),
             'road':(128, 64, 128),
             'sidewalk':(0, 0, 192),
             'people':(0, 64, 128),
             'bicycle': (0, 128, 192)}


image_regions = {128000000: 'building',
             777: 'unknown',
             128128128: 'sky',
             64000128: 'car',
             128064128: 'road',
             192: 'sidewalk',
             64128: 'people',
             128192: 'bicycle'}


for object in os.listdir('c:/Games/anaconda/work/super/Peta/Leuven/my_project/dataset/img'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/Peta/Leuven/Images/Left/2/' + name + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(unique_color(image))
    image = np.array(image, dtype=np.uint32)

    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    new_mask = new_mask + (image[:, :, 0] * 1000000) + (image[:, :, 1] * 1000) + image[:, :, 2]
    new_mask = np.where(new_mask != 0, new_mask, 777)

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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/Peta/Leuven/my_project/dataset/ann/' + name + '.json')
