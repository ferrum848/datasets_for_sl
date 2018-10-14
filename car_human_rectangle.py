image_class = {'fon' : (1, 1, 1),
             'human':(255, 255, 255)}


for object in os.listdir('c:/Games/anaconda/work/super/Peta/HumanCarOmniDataset/Human Set/my_project/dataset/img'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/Peta/HumanCarOmniDataset/Human Set/my_project/dataset/img/' + name + '.jpg')
    all_variables = scipy.io.loadmat('c:/Games/anaconda/work/super/Peta/HumanCarOmniDataset/Human Set/annot/' + name + '.mat')

    foto_objects = []
    json_for_image = {'tags': [],
                      'description': '',
                      'objects': foto_objects,
                      'size': {
                          'width': image.shape[1],
                          'height': image.shape[0]
                      }}

    for i in all_variables['Objects']:
        temp = {"bitmap": "",
                "type": "rectangle",
                "classTitle": 'human',
                "description": "",
                "tags": [],
                "points": {"interior": [], "exterior": [[int(i[0]), int(i[1])], [int(i[2]), int(i[3])]]}}
        foto_objects.append(temp)
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/Peta/HumanCarOmniDataset/Human Set/my_project/dataset/ann/' + name + '.json')
