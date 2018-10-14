
colors =  [[178, 77, 0], [0, 0, 134], [152, 0, 0], [124, 178, 47], [178, 21, 0], [0, 0, 0], [0, 214, 255], [0, 102, 178], [0, 0, 98], [0, 146, 178], [178, 124, 0], [0, 94, 178], [0, 124, 178], [161, 0, 0], [0, 168, 175], [54, 178, 118], [134, 0, 0], [86, 178, 86], [156, 178, 15], [178, 85, 0], [0, 131, 178], [0, 65, 178], [0, 0, 89], [178, 117, 0], [0, 28, 178], [3, 175, 169], [175, 156, 0], [107, 0, 0], [178, 0, 0], [178, 140, 0], [47, 178, 124], [0, 0, 116], [178, 148, 0], [0, 116, 178], [41, 178, 131], [178, 101, 0], [0, 0, 107], [89, 0, 0], [15, 178, 156], [28, 178, 143], [73, 178, 99], [105, 178, 67], [178, 45, 0], [163, 172, 9], [0, 0, 161], [0, 6, 170], [0, 43, 178], [67, 178, 105], [0, 153, 178], [0, 80, 178], [116, 0, 0], [0, 72, 178], [60, 178, 111], [178, 5, 0], [0, 0, 143], [178, 29, 0], [0, 35, 178], [92, 178, 79], [99, 178, 73], [0, 109, 178], [125, 0, 0], [178, 37, 0], [178, 69, 0], [0, 58, 178], [178, 109, 0], [170, 0, 0], [143, 178, 28], [0, 87, 178], [178, 61, 0], [137, 178, 35], [178, 93, 0], [0, 13, 178], [118, 178, 54], [178, 13, 0], [178, 132, 0], [22, 178, 150], [0, 0, 125], [0, 0, 152], [0, 138, 178], [98, 0, 0], [143, 0, 0], [0, 21, 178], [9, 178, 163], [35, 178, 137], [131, 178, 41], [169, 164, 3], [178, 53, 0], [111, 178, 60], [0, 50, 178], [79, 178, 92], [150, 178, 22], [0, 160, 178]]
print(len(colors))
image_class = {}
for i in range(len(colors)):
    image_class['obj' + str(i)] = colors[i]
print(image_class)
image_regions = {178077000: 'obj0', 134: 'obj1', 152000000: 'obj2', 124178047: 'obj3', 178021000: 'obj4', 777: 'obj5', 214255: 'obj6', 102178: 'obj7', 98: 'obj8', 146178: 'obj9', 178124000: 'obj10', 94178: 'obj11', 124178: 'obj12', 161000000: 'obj13', 168175: 'obj14', 54178118: 'obj15', 134000000: 'obj16', 86178086: 'obj17', 156178015: 'obj18', 178085000: 'obj19', 131178: 'obj20', 65178: 'obj21', 89: 'obj22', 178117000: 'obj23', 28178: 'obj24', 3175169: 'obj25', 175156000: 'obj26', 107000000: 'obj27', 178000000: 'obj28', 178140000: 'obj29', 47178124: 'obj30', 116: 'obj31', 178148000: 'obj32', 116178: 'obj33', 41178131: 'obj34', 178101000: 'obj35', 107: 'obj36', 89000000: 'obj37', 15178156: 'obj38', 28178143: 'obj39', 73178099: 'obj40', 105178067: 'obj41', 178045000: 'obj42', 163172009: 'obj43', 161: 'obj44', 6170: 'obj45', 43178: 'obj46', 67178105: 'obj47', 153178: 'obj48', 80178: 'obj49', 116000000: 'obj50', 72178: 'obj51', 60178111: 'obj52', 178005000: 'obj53', 143: 'obj54', 178029000: 'obj55', 35178: 'obj56', 92178079: 'obj57', 99178073: 'obj58', 109178: 'obj59', 125000000: 'obj60', 178037000: 'obj61', 178069000: 'obj62', 58178: 'obj63', 178109000: 'obj64', 170000000: 'obj65', 143178028: 'obj66', 87178: 'obj67', 178061000: 'obj68', 137178035: 'obj69', 178093000: 'obj70', 13178: 'obj71', 118178054: 'obj72', 178013000: 'obj73', 178132000: 'obj74', 22178150: 'obj75', 125: 'obj76', 152: 'obj77', 138178: 'obj78', 98000000: 'obj79', 143000000: 'obj80', 21178: 'obj81', 9178163: 'obj82', 35178137: 'obj83', 131178041: 'obj84', 169164003: 'obj85', 178053000: 'obj86', 111178060: 'obj87', 50178: 'obj88', 79178092: 'obj89', 150178022: 'obj90', 160178: 'obj91'}
#for i, j in image_class.items():
#    image_regions[(j[0] * 1000000) + (j[1] * 1000) + j[2]] = i
#print(image_regions)




for object in os.listdir('c:/Games/anaconda/work/super/Peta/coco/val2017'):
    name = object[:-4]
    print(name)
    image = cv2.imread('c:/Games/anaconda/work/super/Peta/coco/stuff_val2017_pixelmaps/' + name + '.png')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    json_dump(json_for_image, 'c:/Games/anaconda/work/super/Peta/coco/my_project/dataset/ann/' + name + '.json')

