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


pascal_regions = {}
with open('c:/Games/anaconda/work/super/1/labels.txt') as file:
    for line in file:
        line = line.split('\n')[0]
        line = line.split(':')
        pascal_regions[int(line[0])] = line[1][1:]

'''
color_temp, i = [], 1
pascal_classes = {}
while i != 460:
    obj = pascal_regions[i]
    a, b, c = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    if not (a, b, c) in color_temp:
        color_temp.append((a, b, c))
        pascal_classes[obj] = (a, b, c)
        i += 1
print(pascal_classes)
'''




for object in os.listdir('c:/Games/anaconda/work/super/1/my_project/dataset/img'):
    name = object[:11]


    image = cv2.imread('c:/Games/anaconda/work/super/1/my_project/dataset/img/' + name + '.jpg')

    mask = scipy.io.loadmat('c:/Games/anaconda/work/super/1/trainval/' + name + '.mat')['LabelMap']

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}

    for obj in np.unique(mask):
        classTitle = pascal_regions[obj]
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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/1/my_project/dataset/ann/' + name + '.json')

