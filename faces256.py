image_class = {'eyes':(0, 0, 255),
             'nose':(0, 255, 255),
             'mouth':(0, 255, 0),
             'background':(255, 0, 0)}


image_regions = {255: 'eyes',
             255255: 'nose',
             255000: 'mouth',
             255000000: 'background'}


result = []
with open ("training.txt", "r") as file:
    all_lines = file.readlines()
    for line in all_lines:
        line = line.split('\n')[0].split(' ')[1:12]
        line[0] = line[0][9:]
        for i in range(1, 11):
            temp = line[i][:-7]
            line[i] = int(temp)
        result.append(line)
lfw = result[:4151]
#print(lfw)

for obj in lfw:
    im = cv2.imread('c:/Games/anaconda/work/super/faces256/lfw_5590/' + obj[0])
    #shutil.copy('c:/Games/anaconda/work/super/faces256/lfw_5590/' + obj[0], 'c:/Games/anaconda/work/super/faces256/my_project/dataset/img/')
    image = np.ones(im.shape, dtype=np.uint8)
    cv2.ellipse(image, (obj[1], obj[6]), (10, 4), 0, 0, 360, (0, 0, 255), -1)
    cv2.ellipse(image, (obj[2], obj[7]), (10, 4), 0, 0, 360, (0, 0, 255), -1)
    cv2.ellipse(image, (obj[3], obj[8]), (11, 4), 0, 0, 360, (0, 255, 255), -1)
    cv2.ellipse(image, ((obj[4] + obj[5])//2, (obj[10] + obj[9])//2), (abs((obj[5] - obj[4]))//2, abs((obj[10] - obj[9]))//2), 0, 0, 360, (0, 255, 0), -1)



#=====================================================

    name = obj[0][:-4]

    image = np.array(image, dtype=np.uint32)
    print(unique_color(image))

    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    new_mask = new_mask + (image[:, :, 0] * 1000000) + (image[:, :, 1] * 1000) + image[:, :, 2]
    new_mask = np.where(new_mask != 1001001, new_mask, 255000000)

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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/faces256/my_project/dataset/ann/' + name + '.json')

