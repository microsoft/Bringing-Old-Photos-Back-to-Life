# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import struct
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                #print(fname)
                path = os.path.join(root, fname)
                images.append(path)

    return images

### Modify these 3 lines in your own environment
indir="/home/ziyuwan/workspace/data/temp_old"
target_folders=['VOC','Real_L_old','Real_RGB_old']
out_dir ="/home/ziyuwan/workspace/data/temp_old"
###

if os.path.exists(out_dir) is False:
    os.makedirs(out_dir)

#
for target_folder in target_folders:
    curr_indir = os.path.join(indir, target_folder)
    curr_out_file = os.path.join(os.path.join(out_dir, '%s.bigfile'%(target_folder)))
    image_lists = make_dataset(curr_indir)
    image_lists.sort()
    with open(curr_out_file, 'wb') as wfid:
        # write total image number
        wfid.write(struct.pack('i', len(image_lists)))
        for i, img_path in enumerate(image_lists):
             # write file name first
             img_name = os.path.basename(img_path)
             img_name_bytes = img_name.encode('utf-8')
             wfid.write(struct.pack('i', len(img_name_bytes)))
             wfid.write(img_name_bytes)
    #
    #             # write image data in
             with open(img_path, 'rb') as img_fid:
                 img_bytes = img_fid.read()
             wfid.write(struct.pack('i', len(img_bytes)))
             wfid.write(img_bytes)

             if i % 1000 == 0:
                 print('write %d images done' % i)