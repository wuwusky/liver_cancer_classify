import cv2
import pydicom
import os
import numpy as np

#------------------------------------------------------------------------------------
# 遍历训练集或者测试集，讲所有dcm格式影像按照命名顺序拼接组成一个numpy的3维数组，
# 然后保存为npy格式。之所以用命名顺序不使用dicom读取切片DCM文件坐标信息是因为经
# 过数据分析发现部分样本的坐标信息存在问题，不能正确的表示切片的顺序。
#------------------------------------------------------------------------------------
files_dir = '/data/data_for_ai/test/'
files_dir_save  = '/data/data_for_ai/train_temp/'
files_dir_save_npy = '/data/data_for_ai/test_temp_npy/'
if not os.path.exists(files_dir_save):
    os.makedirs(files_dir_save)

# if not os.path.exists(files_dir_save_npy):
#     os.makedirs(files_dir_save_npy)

list_sample_names = os.listdir(files_dir)

for i in range(len(list_sample_names)):

    # if not os.path.exists(files_dir_save + list_sample_names[i]):
    #     os.makedirs(files_dir_save + list_sample_names[i])

    sample_single_dir = files_dir + list_sample_names[i] + '/'
    list_pics = os.listdir(sample_single_dir)
    temp_cts = []

    pic_single_name_half = list_pics[0][:-9]
    for j in range(len(list_pics)):
        temp_pic_name_ID = str(j+1)
        if len(temp_pic_name_ID)  == 3:
            temp_pic_name_ID = '00' + temp_pic_name_ID
        if len(temp_pic_name_ID)  == 2:
            temp_pic_name_ID = '000' + temp_pic_name_ID
        if len(temp_pic_name_ID)  == 1:
            temp_pic_name_ID = '0000' + temp_pic_name_ID
        if len(list_pics[0]) < 14:
            try:
                temp_ct = pydicom.read_file(sample_single_dir + 'image-' + str(j+1) + '.dcm')
            except Exception:
                print('Error')
        else:
            try:
                temp_ct = pydicom.read_file(sample_single_dir + pic_single_name_half + temp_pic_name_ID + '.dcm')
            except Exception:
                print('Error')
        temp_cts.append(temp_ct)

    # print('ok')

    image_cts_np = np.zeros((len(temp_cts), 512, 512),dtype=np.int16)

    # print(temp_cts[0].ImageOrientationPatient[5])
    # if temp_cts[0].ImageOrientationPatient[5] != 0:
    #     print(list_sample_names[i])
    for k in range(len(temp_cts)):

        # print(temp_cts[k].PatientPosition)
        # print(temp_cts[k].SliceThickness)
        temp_image = temp_cts[k].pixel_array
        temp_image.astype(np.int16)
        temp_image[temp_image < 0] = 0
        temp_image[temp_image > 4096] = 0
        intercept = temp_cts[k].RescaleIntercept
        slope = temp_cts[k].RescaleSlope

        if slope != 1:
            temp_image = slope * temp_image.astype(np.float64)
            temp_image = temp_image.astype(np.int16)
        temp_image = temp_image + intercept

        image_cts_np[k,:,:] = temp_image
        # cv2.imshow('temp', temp_image)
        # cv2.waitKey()
    #     cv2.imwrite(files_dir_save + list_sample_names[i] + '/' + str(k) + '.png', temp_image)
    np.save(files_dir_save_npy + list_sample_names[i] + '.npy', image_cts_np)




    print(i, 'ok')

