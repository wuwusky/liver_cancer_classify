# liver_cancer_classify
liver cancer classify model with DL(3D-Conv)
参加大数据医疗—肝癌影像AI诊断比赛，最高得分0.801

环境：
python 3.5以上
pytorch 1.0
opencv
pydicom
numpy
matplotlib
scipy


运行：
首先运行dcm2pic.py，分别修改对应的训练集和测试集路径以及其对应的保存路径
然后运行test_Conv3d_pic.py，注意修改训练数据和测试数据路径（保存为npy格式的路径）
训练模型：更改变量train_flag = True
预测模式：更改变量train_flag = False
