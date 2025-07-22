import os.path

from torch.utils.data import Dataset

from  PIL import Image
import numpy as np
import pandas as pd
import re

from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, args, is_train):
        """
        args.root_directory :  /Users/image_dateset
        args.train_list : train_data_list_k20.txt
        args.train_label_list: train_label_list_r5.txt
        """

        # usually we need args rather than single datalist to init the dataset
        super().__init__()

        root_directory = args.root_directory  # 数据根目录
        if is_train:
            data_list = os.path.join(root_directory,
                                     args.train_list)  # ex: /Users/image_dateset , train_data_list_k20.txt
            label_list = os.path.join(root_directory,
                                      args.train_label_list)  # ex: /Users/image_dateset,train_label_list_r5.txt
        else:
            data_list = os.path.join(root_directory, args.val_list)
            label_list = os.path.join(root_directory, args.val_label_list)

        """"
        读取图片路径以及图片ID
        """
        # read images paths 读取样本图片的地址以及，样本图片的image_id
        images = [line.split() for line in open(data_list).readlines()  ]
        self.img_paths = [os.path.join(root_directory, info[0]) for info in images]  # ex: [  /Users/20230101
        # /20230101_000001_k20.png]
        self.image_ids = [info[1] for info in images]

        """
        读取标签地址，并将标签数据加载到pandas 中
        
        """
        image_label_paths = [line.split() for line in open(label_list).readlines()   ]
        img_label_paths = [ os.path.join(root_directory, info[0]) for info in image_label_paths]
        data_frames = [pd.read_csv(img_label_path) for img_label_path in img_label_paths]
        self.image_labels = pd.concat(data_frames, ignore_index=True).set_index('image_id')
        self.transform = transforms.Compose([
            transforms.Resize((500, 200)),  # 根据需要调整图像大小
           # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
            ,transforms.Normalize([0.5], [0.5])  # 标准归一化, p1.均值  p2.方差
        ])

    def process(self, img):
        # cv: h, w, c, tensor: c, h, w
        img = img.transpose((2, 0, 1)).astype(np.float32)
        # you can add other process method or augment here
        return img

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        file_path = self.img_paths[idx]
        img = Image.open(file_path).convert('RGB')
        file_name_with_extension = os.path.basename(file_path)
        # 分割文件名和后缀
        image_id, extension = os.path.splitext(file_name_with_extension)
        image_id = re.sub(r'_k60$', '', image_id)
        # 根据image的后缀，读取标签
        img = self.transform(img)

        label = self.image_labels.loc[image_id, "label"]
        label = int(label)

        return img, label
