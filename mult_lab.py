p_class = {'background' : (143, 0, 0),
             'hair':(255, 32, 0),
             'face':(255, 191, 0),
             'upperclothes':(159, 255, 96),
             'lowerclothes':(0, 255, 255),
             'arms':(0, 80, 255),
             'shoes':(0, 0, 143),
             }


p_regions = {143000000: 'background',
             255032000: 'hair',
             255191000: 'face',
             159255096: 'upperclothes',
             255255: 'lowerclothes',
             80255: 'arms',
             143: 'shoes',
             }


for object in os.listdir('c:/Games/anaconda/work/super/7/my_project/dataset/img'):
    name = object[:-4]

    image = cv2.imread('c:/Games/anaconda/work/super/7/my_project/dataset/img/' + name + '.jpg')
    im = cv2.imread('c:/Games/anaconda/work/super/7/annotation/' + name + '_m.png')
    print(unique_color(im))

    im = cv2.resize(im, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)
    im = np.array(im, dtype=np.uint32)

    new_mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint32)
    new_mask = new_mask + (im[:, :, 0] * 1000000) + (im[:, :, 1] * 1000) + im[:, :, 2]
    new_mask = np.where(new_mask != 175, new_mask, 143)

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}


    for obj in np.unique(new_mask):
        classTitle = p_regions[obj]
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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/7/my_project/dataset/ann/' + name + '.json')

