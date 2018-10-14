res = {}
with open('c:/Games/anaconda/work/super/new/fgvc-aircraft-2013b/data/images_box.txt') as file:
    for line in file:
        line = line.split('\n')[0]
        line = line.split(' ')
        res[line[0]] = line[1:]
print(res)




image_class = {'fon' : (1, 1, 1),
             'plane':(0, 0, 255)}

classes = []
for title, color in image_class.items():
    temp = {'title': title, 'shape': 'rectangle', 'color': color2code(color)}
    classes.append(temp)
meta = {'classes': classes, 'tags_images': [], "tags_objects": []}
json_dump(meta, 'c:/Games/anaconda/work/super/new/fgvc-aircraft-2013b/my_project/meta.json')


for object in os.listdir('c:/Games/anaconda/work/super/new/fgvc-aircraft-2013b/data/images'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/new/fgvc-aircraft-2013b/data/images/' + name + '.jpg')

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}

    temp = {"bitmap": "",
                "type": "rectangle",
                "classTitle": 'plane',
                "description": "",
                "tags": [],
                "points": {"interior": [], "exterior": [[int(res[name][0]), int(res[name][1])], [int(res[name][2]), int(res[name][3])]]}}
    foto_objects.append(temp)
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/new/fgvc-aircraft-2013b/my_project/dataset/ann/' + name + '.json')


