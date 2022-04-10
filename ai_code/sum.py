#coding: utf-8
import pyaudio 
# from pyaudio import PyAudio, paInt16
import numpy as np
from datetime import datetime
import wave

#获得音频
input_name = 'input.wav'
input_stress = 'H:\\作业\\专业设计\\大三下\\专业设计\\ai+healthcare\\ai_code'
input_path = input_stress + input_name
#将麦克风的输出存储
def get_audio(filepath):
    aa = str(input("是否开始录音？   （是/否）"))
    if aa == str("是") :
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                # 声道数
        RATE = 11025                # 采样率
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = filepath
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("*"*10, "开始录音：请在5秒内输入语音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("*"*10, "录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    elif aa == str("否"):
        exit()
    else:
        print("无效输入，请重新选择")

get_audio(input_path)

audio_f = open(input_path, mode='r')
#语音识别转文本

from speechTotext import speechtotext
audio_text = speechtotext(audio_f)

#判断文本情感
from text_emotion import Text_emotion
txt_mood,txt_score = Text_emotion(audio_text)

#判断语音情感 ！或许 ok
#///from speech_emotion_master import *
from speech_emotion_recognition_v02 import *
model_se, accuracy_score = model_train()
pred_se = myaudio(input_path, model_se)
pred_score = pred_se * 10 * accuracy_score
#将语音情感和文本情感比对，得出音频的情感变化
#/// compare pred_se & txt_mood;  pred_score & txt_score


#提炼关键词，有词的出现频次排序，词的关联次数排序
import jieba
from jieba.analyse import *
import jieba.posseg as pseg
from speech_keyword import keywords
keywords(audio_text)

#根据数据库前置生成basic 音乐 ！ok

#根据语音情感和语音数据影响生成音乐的旋律变化走向! ok
#得到初级的midi文件

#根据提炼的关键词找出名词，形容词，拟声词…… ！  
#根据进一步提炼的关键词在数据库中寻找音频元素 ！

#根据词的比重、出现次数、关联次数安排叠合、插入，得到二级的midi文件

#以二级midi文件/或者分析乐理，生成个性化实时音乐 ！
import musicpy 


