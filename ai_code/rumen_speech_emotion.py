import os
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import hmmlearn.hmm as hl
def search_file(directory):
    """
    :param directory: 训练音频的路径
    :return: 字典{'apple':[url, url, url ... ], 'banana':[...]}
    """
    # 使传过来的directory匹配当前操作系统
    directory = os.path.normpath(directory)
    objects = {}
    # curdir：当前目录
    # subdirs: 当前目录下的所有子目录
    # files: 当前目录下的所有文件名
    for curdir, subdirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                label = curdir.split(os.path.sep)[-1]   # os.path.sep为路径分隔符
                if label not in objects:
                    objects[label] = []
                # 把路径添加到label对应的列表中
                path = os.path.join(curdir, file)
                objects[label].append(path)
    return objects
train_samples = search_file('G:/Google download/CASIA database/liuchanhg')#需要修改为自己的训练库
#print(train_samples)
train_x, train_y = [], []
# 遍历字典
for label, filenames in train_samples.items():
    # [('apple', ['url1,,url2...'])
    # [("banana"),("url1,url2,url3...")]...
    mfccs = np.array([])
    for filename in filenames:
        sample_rate, sigs = wf.read(filename)
        mfcc = sf.mfcc(sigs, sample_rate)
        if len(mfccs) == 0:
            mfccs = mfcc
        else:
            mfccs = np.append(mfccs, mfcc, axis=0)
    train_x.append(mfccs)
    train_y.append(label)
    models = {}
for mfccs, label in zip(train_x, train_y):
    model = hl.GaussianHMM(n_components=4, covariance_type='diag', n_iter=1000)
    models[label] = model.fit(mfccs)  # # {'apple':object, 'banana':object ...}
    # 读取测试集数据
test_samples = search_file('G:/Google download/CASIA database/wangzhe')#需要修改为自己的测试库
test_x, test_y = [], []
for label, filenames in test_samples.items():
    mfccs = np.array([])
    for filename in filenames:
        sample_rate, sigs = wf.read(filename)
        mfcc = sf.mfcc(sigs, sample_rate)
        if len(mfccs) == 0:
            mfccs = mfcc
        else:
            mfccs = np.append(mfccs, mfcc, axis=0)
    test_x.append(mfccs)
    test_y.append(label)
    pred_test_y = []
for mfccs in test_x:
    # 判断mfccs与哪一个HMM模型更加匹配
    best_score, best_label = None, None
    # 遍历7个模型
    for label, model in models.items():
        score = model.score(mfccs)
        if (best_score is None) or (best_score < score):
            best_score = score
            best_label = label
    pred_test_y.append(best_label)
 
print(test_y)   
print(pred_test_y)