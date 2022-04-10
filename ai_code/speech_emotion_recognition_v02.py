#就是说，是需要先将音频的特征处理处理成能够被程序读取的格式的。

#导入相关库
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#定义函数提取音频特征
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

#定义字典保存音频情绪
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'fearful', 'disgust']#可以修改测试的情绪总类

#加载数据位置
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("H:\\作业\\专业设计\\大三下\\专业设计\\ai+healthcare\\ai_code\\语音情绪识别训练\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        #H:\\作业\\专业设计\\大三下\\专业设计\\ai+healthcare\\ai_code\\语音情绪识别训练\\speech-emotion-recognition-ravdess-data
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def model_train():
    #将数据集划分为训练和测试两个部分
    x_train,x_test,y_train,y_test=load_data(test_size=0.25)

    #观察数据集的音频形状
    print((x_train.shape[0], x_test.shape[0]))

    #获取提取的特征数量
    print(f'Features extracted: {x_train.shape[1]}')

    #初始化分类的神经网络模型
    model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

    #拟合数据，训练模型
    model.fit(x_train,y_train)

    #测试，输出训练后的准确性
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy*100))

    return model, accuracy


#测试我的音频
def myaudio(audio, model):
    extract_feature(audio)
    audio_pred = model.predict(audio)
    print(audio_pred)
    return audio_pred