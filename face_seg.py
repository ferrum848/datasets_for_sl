image_class = {'skin' : (255, 255, 0),
             'hair':(127, 0, 0),
             'eyes':(0, 0, 255),
             'nose':(0, 255, 255),
             'mouth':(0, 255, 0),
             'background':(255, 0, 0)}


image_regions = {255255000: 'skin',
             127000000: 'hair',
             255: 'eyes',
             255255: 'nose',
             255000: 'mouth',
             255000000: 'background'}


for object in os.listdir('c:/Games/anaconda/work/super/fasseg/my_project/dataset/img'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/fasseg/V1/Test_Labels/' + name + '.bmp')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.uint32)

    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    new_mask = new_mask + (image[:, :, 0] * 1000000) + (image[:, :, 1] * 1000) + image[:, :, 2]
    #new_mask = np.where(new_mask != 0, new_mask, 111)

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}

    for obj in image_regions.keys():
        classTitle = image_regions[obj]
        mask_bool = np.where(new_mask == obj, new_mask, False)
        print(obj)
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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/fasseg/my_project/dataset/ann/' + name + '.json')


