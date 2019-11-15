import json
import os
import os.path as osp
import sys


# if platform == 'philly':
#     imgPathDict =  {'COCO_trainval': None,
#                     'coco_2014_train': 'coco/train2014/',
#                     'coco_2014_valminusminival': 'coco/val2014/',
#                     'furniture_train': 'HomeFurniture/Images.zip@/Images/',
#                     'FashionV2_train': 'FashionV2/Images.zip@/Images/',
#                     'OpenImage_train': 'OpenImage/train_images.zip@/train_images/'
#                     }
# else:
#     imgPathDict =  {'COCO_trainval': None,
#                     'coco_2014_train': 'coco/train2014/',
#                     'coco_2014_valminusminival': 'coco/val2014/',
#                     'furniture_train': 'HomeFurniture/Images/',
#                     'FashionV2_train': 'FashionV2/Images/',
#                     'OpenImage_train': 'OpenImage/train_images/'
#                     }


jsonFile = sys.argv[1]

outFile = jsonFile.replace('.json', '_philly.json')

print ("outFile = {}".format(outFile))

imgPathMap =  {'HomeFurniture': 'HomeFurniture/Images.zip@',
                'FashionV2': 'FashionV2/Images.zip@',
                'OpenImage/train_images/': 'OpenImage/train_images.zip@/train_images/'
                }


anno = json.load(open(jsonFile,'r'))

for ix, img in enumerate(anno['images']):
    org = img['file_name']
    for k,v in imgPathMap.items():
        if k in img['file_name']:
            img['file_name'] = img['file_name'].replace(k,v)
            break
    if (ix+1)%100 == 0:
        print ("{} ==>\n{}".format(org, anno['images'][ix]['file_name']))
               
print ("Saving annotatinos to {}".format(outFile))
with open(outFile, 'w') as fout:
    json.dump(anno, fout)
print ("Done")
          

        

