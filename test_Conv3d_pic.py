import os
import torch
# import SimpleITK as sitk
from torch.utils.data import DataLoader
import numpy as np
import csv
import models_3d as mm
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#截取中央部分的样本块
def get_crop_from_np(img_np, out_size):
    d, h, w = img_np.shape
    new_d, new_h, new_w = out_size

    front = int((d-new_d)/2)
    top = np.random.randint(0, h-new_h)
    left = np.random.randint(0, w-new_w)

    img_np_crop = img_np[front:front+new_d, top:top+new_h, left:left+new_w]

    return img_np_crop
#截图z轴的中央部分，x和y轴随机截取，组成一个样本块
def get_crop_from_np_ct(img_np, out_size):
    d, h, w = img_np.shape
    new_d, new_h, new_w = out_size

    front = int((d-new_d)/2)
    top = int((h-new_h)/2)
    left = int((w-new_w)/2)

    img_np_crop_ct = img_np[front:front+new_d, top:top+new_h, left:left+new_w]

    return img_np_crop_ct
#直接读取dcm文件
def get_np_from_dcm(file_dir, flag_crop=True):

    file_names = os.listdir(file_dir)
    h_ct = len(file_names)
    image_np = np.zeros((h_ct, 512, 512))
    for i in range(len(file_names)):
        temp_np = cv2.imread(file_dir + '/' + str(i) + '.png')
        temp_np = cv2.cvtColor(temp_np, cv2.COLOR_BGR2GRAY)
        image_np[i,:,:] = temp_np
    image_np = image_np.astype('float')
    # temp_mean = np.mean(image_np)
    # temp_max = image_np.max()
    # temp_min = image_np.min()
    # image_np = (image_np-temp_mean)/(temp_max - temp_min)
    # slope = image.RescaleSlope
    # intercept = image.RescaleIntercept


    if flag_crop:
        image_np = np.resize(image_np, (image_np.shape[0], 80, 80))
        image_np = np.resize(image_np, (40, 70, 70))
        image_np_ex = get_crop_from_np(img_np=image_np, out_size=(32, 64, 64))


        return image_np_ex
    else:
        image_np = np.resize(image_np, (image_np.shape[0], 80, 80))
        image_np = np.resize(image_np, (40, 70, 70))
        image_np_ct = get_crop_from_np_ct(img_np=image_np, out_size=(32, 64, 64))
        return image_np_ct
#读取转换保存好的npy数据，恢复成numpy三维数组
def get_np_from_npy(file_dir, flag_crop=True):
    image_np = np.load(file_dir)
    image_np = image_np.astype(np.uint8)
    MIN_BOUND = 0
    MAX_BOUND = 200
    image_np[image_np < MIN_BOUND] = 0
    image_np[image_np > MAX_BOUND] = 0

    flag_flip = random.randint(-1, 2)

    for i in range(image_np.shape[0]):
        img_temp = image_np[i, :, :]
        img_temp = cv2.equalizeHist(img_temp)#利用均衡化进行对比度拉伸
        if flag_crop:
            if flag_flip < 2:
                img_temp = cv2.flip(img_temp, flag_flip)#随机进行镜像翻转
        image_np[i, :, :] = img_temp

        f_scale_0 = float(128) / float(image_np.shape[0])
        f_scale_1 = float(150) / float(image_np.shape[1])
    if flag_crop:
        image_np = scipy.ndimage.zoom(input=image_np, zoom=[f_scale_0, f_scale_1, f_scale_1])#更改尺寸
        image_np_ex = get_crop_from_np(img_np=image_np, out_size=(64, 128, 128))

    else:
        image_np = scipy.ndimage.zoom(input=image_np, zoom=[f_scale_0, f_scale_1, f_scale_1])
        image_np_ex = get_crop_from_np_ct(img_np=image_np, out_size=(64, 128, 128))

    return image_np_ex
#计算训练和验证集合的正确率
def get_acc(data_loader, model):
    correct = 0
    total = 0
    model.eval()
    for i,(img, labels) in enumerate(data_loader):
        img = img.to(device)
        labels = labels.to(device)
        outputs = model(img)
        labels = labels.squeeze_()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print(total)
    acc_rate = 100 * float(correct) / float(total)

    return acc_rate
#样本类，得到每次取样的样本张量、标签张量
class my_dataset():
    def __init__(self, list_filenames, data_dir, dict_label, flag_crop=True):
        self.list_filenames = list_filenames
        self.data_dir = data_dir
        self.dict = dict_label
        self.flag_crop = flag_crop

    def __getitem__(self, item):
        img_name = self.list_filenames[item]
        img_np = get_np_from_npy(self.data_dir + img_name, flag_crop=self.flag_crop)
        img_tensor = torch.Tensor(img_np)
        img_tensor = torch.stack([img_tensor], 0)

        label = self.dict[img_name[:-4]]

        label_tensor = torch.from_numpy(np.array([int(label)], dtype=np.float32))
        label_tensor = label_tensor.long()
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.list_filenames)

train_flag = False #确定是否进行训练流程，True-进行训练，False-根据训练保存模型进行前向预测

data_dir = '/data/data_for_ai/train_all/' #指定训练集合路径
list_file_names = os.listdir(data_dir)
data_submit_dir = '/data/data_for_ai/test_temp_npy/' #废弃（原来利用leak的测试集做验证）
list_file_names_submit = os.listdir(data_submit_dir)

model_dir = 'test_128_32_3D_res_pic.pb' #模型保存路径和名字

label_dir = './train_label_all.csv' #训练样本标签
list_label = csv.reader(open(label_dir, 'r'))
dict_label = {'0':'0'}
for row in list_label:
    dict_label[row[0]] = row[1]

label_submit_dir = './submit.csv' #原来leak的测试集标签
list_label_submit = csv.reader(open(label_submit_dir, 'r'))
dict_label_submit = {'0':'0'}
for row in list_label_submit:
    dict_label_submit[row[0]] = row[1]

num_train = int(len(list_file_names) * 0.5) #指定训练集和验证集的占比
random.shuffle(list_file_names)
random.shuffle(list_file_names)
list_file_names_train = []
list_file_names_test = []
list_file_names_train = list_file_names[:num_train]
list_file_names_test = list_file_names[num_train:]

n_batch_size = 24
n_show_step = 10

train_data = my_dataset(list_filenames=list_file_names_train, data_dir=data_dir, dict_label=dict_label, flag_crop=True)
train_data_loader = DataLoader(train_data, shuffle=True, batch_size=n_batch_size, num_workers=64)

test_data = my_dataset(list_filenames=list_file_names_test, data_dir=data_dir, dict_label=dict_label, flag_crop=False)
test_data_loader = DataLoader(test_data, shuffle=True, batch_size=n_batch_size, num_workers=64)

data_all = my_dataset(list_filenames=list_file_names, data_dir=data_dir, dict_label=dict_label, flag_crop=True)
data_all_loader = DataLoader(data_all, shuffle=True, batch_size=n_batch_size, num_workers=64)

data_submit = my_dataset(list_filenames=list_file_names_submit, data_dir=data_submit_dir, dict_label=dict_label_submit, flag_crop=False)
data_submit_loader = DataLoader(data_submit, shuffle=True, batch_size=n_batch_size, num_workers=64)
epoch_max = 150


list_loss = []
list_rate = []
list_rate_submit = []
model = mm.ResNet(mm.ResidualBlock, [2, 2, 2, 2]) #采用ResNet-18的结构
if train_flag:
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    total_step_train = len(train_data_loader)
    total_step_all = len(data_all_loader)
    for epoch in range(epoch_max):
        if epoch == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        if epoch < 100:
            for i, (img, labels) in enumerate(train_data_loader):
                model.train()
                img = img.to(device)
                labels = labels.to(device)
                output = model(img)
                labels = labels.squeeze_()
                loss_end = criterion(output, labels)

                optimizer.zero_grad()
                loss = loss_end
                loss.backward()
                optimizer.step()

                if (i+1) % n_show_step == 0:
                    list_loss.append(loss.item())
                    print('Epoch[{}/{}], Step[{}/{}] Loss:{:.7f}'.format(epoch + 1, epoch_max, i + 1, total_step_train, loss.item()))
        else:
            for i, (img, labels) in enumerate(data_all_loader):
                model.train()
                img = img.to(device)
                labels = labels.to(device)
                output = model(img)
                labels = labels.squeeze_()
                loss_end = criterion(output, labels)

                optimizer.zero_grad()
                loss = loss_end
                loss.backward()
                optimizer.step()

                if (i + 1) % n_show_step == 0:
                    list_loss.append(loss.item())
                    print('Epoch[{}/{}], Step[{}/{}] Loss:{:.7f}'.format(epoch + 1, epoch_max, i + 1, total_step_all, loss.item()))
        rate_temp = get_acc(test_data_loader, model)
        # rate_submit = get_acc(data_submit_loader, model)
        list_rate.append(rate_temp)
        # list_rate_submit.append(rate_submit)

        print('rate:{:.2f}%  rate_submit:{:.2f}%'.format(rate_temp, 0))

    fig_1 = plt.figure(1)
    line= plt.plot(list_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.ylim(top = 1)
    plt.savefig('./loss.png')
    # plt.show()
    # plt.close()

    fig_2 = plt.figure(2)
    line = plt.plot(list_rate)
    # line_2 = plt.plot(list_rate_submit)
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.ylim(top = 100)
    plt.savefig('./rate.png')
    plt.show()
    plt.close()



    model.eval()
    torch.save(model.state_dict(), model_dir)
else:
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(model_dir))
    model.eval().to(device)

    # temp_rate = get_acc(train_data_loader, model)
    # print(temp_rate)
    data_dir_test = '/data/data_for_ai/test_data_new_npy/'
    list_file_names = os.listdir(data_dir_test)
    import datetime
    date = datetime.datetime.now()
    result_dir = './ret_' + str(date.year) + '-' + str(date.month) + '-' + str(date.day) + '-' + str(date.hour) + '.csv'
    ret_file = open(result_dir, 'a', newline='')
    ret_write = csv.writer(ret_file, dialect='excel')

    for i in range(len(list_file_names)):
        list_temp_ret = []
        test_data_tensor = torch.Tensor(get_np_from_npy(data_dir_test+list_file_names[i], flag_crop=False))
        test_data_tensor = torch.stack([test_data_tensor], 0).to(device)
        test_data_tensor = torch.stack([test_data_tensor], 0).to(device)

        out = model(test_data_tensor)
        _, value_predict = torch.max(out.data, 1)

        list_temp_ret.append(list_file_names[i][:-4])
        list_temp_ret.append(int(value_predict))
        print(i, len(list_file_names))
        ret_write.writerow(list_temp_ret)



