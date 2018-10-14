image_class = {'background' : (0, 0, 0),
             'head':(128, 0, 0),
             'right torso':(0, 128, 0),
             'left upper arm':(128, 128, 0),
             'right upper arm':(0, 0, 128),
             'left upper leg':(128, 0, 128),
             'right upper leg':(0, 128, 128),
             'left torso':(128, 128, 128),
             'left lower arm':(64, 0, 0),
             'right lower arm':(192, 0, 0),
             'left lower leg':(64, 128, 0),
             'right lower leg':(192, 128, 0),
             'body part not labelled':(255, 255, 255)
             }


image_regions = {777: 'background',
             128000000: 'head',
             128000: 'right torso',
             128128000: 'left upper arm',
             128: 'right upper arm',
             128000128: 'left upper leg',
             128128: 'right upper leg',
             128128128: 'left torso',
             64000000: 'left lower arm',
             192000000:'right lower arm',
             64128000:'left lower leg',
             192128000:'right lower leg',
             255255255:'body part not labelled'
             }



for object in os.listdir('c:/Games/anaconda/work/super/Peta/Buffy/Buffy/Images2x/buffy_s5e2'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/Peta/Buffy/Buffy/BodyParts2x/buffy_s5e2/' + name + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/Peta/Buffy/my_project/dataset/ann/' + name + '.json')

print(time.time() - start)

