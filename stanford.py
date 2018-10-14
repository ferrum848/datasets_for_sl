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




our_class = {'sky' : (255, 255, 0),
             'tree':(0, 128, 0),
             'road':(0, 0, 255),
             'grass':(128, 255, 0),
             'water':(255, 0, 0),
             'building':(0, 255, 255),
             'mountain':(211, 211, 211),
             'foreground_object':(255, 0, 255),
             'unknown':(0, 0, 0)}


our_regions = {8: 'sky',
             1: 'tree',
             2: 'road',
             3: 'grass',
             4: 'water',
             5: 'building',
             6: 'mountain',
             7: 'foreground_object',
             -1: 'unknown'}

classes = []
for title, color in our_class.items():
    temp = {'title': title, 'shape': 'bitmap', 'color': color2code(color)}
    classes.append(temp)
meta = {'classes': classes, 'tags_images': [], "tags_objects": []}
json_dump(meta, 'c:/Games/anaconda/work/super/4/meta.json')




for object in os.listdir('c:/Games/anaconda/work/super/3/images'):
    name = object[:7]

    image = cv2.imread('c:/Games/anaconda/work/super/3/images/' + name + '.jpg')

    mask, mask_regions = [], []
    with open('c:/Games/anaconda/work/super/3/labels/' + name + '.layers.txt') as file:
        for line in file:
            line = line.split('\n')[0]
            line = line.split(' ')
            mask.append(line)
    mask = np.array(mask, int)

    with open('c:/Games/anaconda/work/super/3/labels/' + name + '.regions.txt') as file:
        for line in file:
            line = line.split('\n')[0]
            line = line.split(' ')
            mask_regions.append(line)
    mask_regions = np.array(mask_regions, int)

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}
    mask = np.where(mask != 0, mask, 135)
    mask_regions = np.where(mask_regions != 0, mask_regions, 8)
    for obj in np.unique(mask):

        xx, yy = [], []
        for index, x in np.ndenumerate(mask):
            if x == obj:
                xx.append(index[0])
                yy.append(index[1])
        midle_coord_color = mask_regions[sum(xx) // len(xx)][sum(yy) // len(yy)]

        classTitle = our_regions[midle_coord_color]

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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/3/ann/' + name + '.json')