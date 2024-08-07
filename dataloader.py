from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as trans
import os
path = os.getcwd()
trans1 = trans.ToTensor()
import torch
from matplotlib import pyplot as plt
class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        super(MyDataset, self).__init__()
        self.path = path

        OK_imgs = []
        OK_img_path = path + "\\OK"
        for img in os.listdir(OK_img_path):
            img = img.rstrip()
            OK_imgs.append(os.path.join(self.path + '\\OK', img))

        NG_imgs = []
        NG_img_path = path + "\\NG"
        for img in os.listdir(NG_img_path):
            img = img.rstrip()
            NG_imgs.append(os.path.join(self.path + '\\NG', img))

        self.OK_imgs = OK_imgs
        self.NG_imgs = NG_imgs
        self.transform = transform

    def __len__(self):
        return len(self.OK_imgs)

    def __getitem__(self, item):
        OK_img_path = self.OK_imgs[item]
        NG_img_path = self.NG_imgs[item]

        OK_img = Image.open(OK_img_path)
        OK_img = OK_img.resize((512,512)).convert('RGB')
        OK_img = trans1(OK_img)

        NG_img = Image.open(NG_img_path).convert('RGB')
        NG_img = NG_img.resize((512, 512))
        NG_img = trans1(NG_img)

        if self.transform is not None:
            OK_img = self.transform(OK_img)
            NG_img = self.transform(NG_img)

        return OK_img, NG_img

path = os.getcwd()
data = MyDataset(path, transform=None)

train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,drop_last=False,num_workers=0)
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,drop_last=False,num_workers=0)

# for i, (images, GT) in enumerate(test_loader):
#     for batch in range(4):
#         print(i)
#         print(GT.shape)
#
#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#
#         swapped_images = images[batch].swapaxes(0, 2).swapaxes(0, 1)
#         axs[0].imshow(swapped_images)
#         axs[0].set_title('OK')
#         axs[0].axis('off')
#
#
#         swapped_GT = GT[batch].swapaxes(0, 2).swapaxes(0, 1)
#         axs[1].imshow(swapped_GT)
#         axs[1].set_title('NG')
#         axs[1].axis('off')
#
#         plt.show()


