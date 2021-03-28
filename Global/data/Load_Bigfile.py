# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import os
import struct
from PIL import Image

class BigFileMemoryLoader(object):
    def __load_bigfile(self):
        print('start load bigfile (%0.02f GB) into memory' % (os.path.getsize(self.file_path)/1024/1024/1024))
        with open(self.file_path, 'rb') as fid:
            self.img_num = struct.unpack('i', fid.read(4))[0]
            self.img_names = []
            self.img_bytes = []
            print('find total %d images' % self.img_num)
            for i in range(self.img_num):
                img_name_len = struct.unpack('i', fid.read(4))[0]
                img_name = fid.read(img_name_len).decode('utf-8')
                self.img_names.append(img_name)
                img_bytes_len = struct.unpack('i', fid.read(4))[0]
                self.img_bytes.append(fid.read(img_bytes_len))
                if i % 5000 == 0:
                    print('load %d images done' % i)
            print('load all %d images done' % self.img_num)

    def __init__(self, file_path):
        super(BigFileMemoryLoader, self).__init__()
        self.file_path = file_path
        self.__load_bigfile()

    def __getitem__(self, index):
        try:
            img = Image.open(io.BytesIO(self.img_bytes[index])).convert('RGB')
            return self.img_names[index], img
        except Exception:
            print('Image read error for index %d: %s' % (index, self.img_names[index]))
            return self.__getitem__((index+1)%self.img_num)


    def __len__(self):
        return self.img_num
