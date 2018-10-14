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

meta_class = {'fon': (255, 0, 0),
              'skin': (0, 255, 255)}

img_json = {1: 'fon',
            255: 'skin'}


for object in os.listdir('c:/Games/anaconda/work/super/2/Pratheepan_Dataset/FamilyPhoto'):
    name = object[:-4]
    image = cv2.imread('c:/Games/anaconda/work/super/2/Pratheepan_Dataset/FamilyPhoto/' + object)


    im = cv2.imread('c:/Games/anaconda/work/super/2/Ground_Truth/GroundT_FamilyPhoto/' + name + '.png')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    split = cv2.split(im)
    mask = split[0]
    mask = np.where(mask >= 255, mask, 1)

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}
    
    classTitle = img_json[1]
    mask_bool = np.where(mask == 1, mask, False)
    left_coner, mask_bool = coords(mask_bool, 1)
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

    classTitle = img_json[255]
    
    for i in range(len(contours)):
        contours[i] = np.array(contours[i], dtype=int)
        if len(contours[i]) > 7:
            min_left = (min(contours[i], key=lambda x: x[0][0])[0][0], min(contours[i], key=lambda x: x[0][1])[0][1])
            max_right = (max(contours[i], key=lambda x: x[0][0])[0][0], max(contours[i], key=lambda x: x[0][1])[0][1])
            srez = mask[min_left[1]: max_right[1] + 1, min_left[0]: max_right[0] + 1]
            mask_bool = np.where(srez == 255, srez, False)
            mask_bool = mask_bool.astype(np.bool)
            data = mask_2_base64(mask_bool)
            temp = {"bitmap":
                        {"origin": [int(min_left[0]), int(min_left[1])],
                         "data": data},
                         "type": "bitmap",
                         "classTitle": classTitle,
                         "description": "",
                         "tags": [],
                         "points": {"interior": [], "exterior": []}}
            foto_objects.append(temp)
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/2/ann/' + name + '.json')
