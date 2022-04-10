import os
import wave
from array import array
from struct import pack
from sys import byteorder

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

import pyaudio
from emotion_recognition import EmotionRecognizer
from utils import get_best_estimators

THRESHOLD = 500    # threshold 临界值
CHUNK_SIZE = 1024  # 数据块尺寸
FORMAT = pyaudio.paInt16
RATE = 16000;      # 速率
SILENCE = 30       # slience 安静

def is_silent(snd_data):  #是否是静音
    "Returns 'True' if below the 'silent' threshold"
    #如果低于静音的阀值，就返回true
    return max(snd_data) < THRESHOLD

def normalize(snd_data):  #正则化
    "Average the volume out"
    # 平均体积(卷)
    MAXIMUM = 16384   #最大量
    times = float(MAXIMUM)/max(abs(i) for i in snd_data) 
    # 遍历正则化。

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    # 修剪开始和结束的空白斑点
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    # 左侧修剪
    snd_data = _trim(snd_data)

    # Trim to the right
    # 右侧修剪
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    # 把沉默添加到 snd_data 和 在长 “秒” 里面。
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    """

  #  *********************
  #  用麦克风，记录下一个或者几个词。
  #  返回短标识.
  # *********************

    """
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """

#*************************
# 正则化处理语音.
# 从开始到结束把静音部分给剪出来.
# 添加 0.5s 的空白.
# 去确定 VLC (多媒体) 能充分发挥功能，而不会被砍掉。
#*************************

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    # 把 麦克风的 记录的数据，存放到 path 路径里。
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}



if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    import argparse
    parser = argparse.ArgumentParser(description="""
                                    Testing emotion recognition system using your voice, 
                                    please consider changing the model and/or parameters as you wish.

                                    ****************
                                    用声音测试情绪识别系统，
                                    请根据您的需要考虑更改模型和/或参数。
                                    *****************
                                    """)
    parser.add_argument("-e", "--emotions", help=
                                            """Emotions to recognize separated by a comma ',', available emotions are
                                            "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise)
                                            and "boredom", default is "sad,neutral,happy"
                                            **************************
                                            情绪用 ， 来分割，可以获得的情绪有  

                                             "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise)
                                            and "boredom", default is "sad,neutral,happy" 

                                            ps 的意思 pleasant surprise 惊喜
                                            **************************
                                            """, default="sad,neutral,happy")
    parser.add_argument("-m", "--model", help=
                                        """
                                        The model to use, 8 models available are: {},
                                        default is "BaggingClassifier"
                                        """.format(estimators_str), default="BaggingClassifier")


    # Parse the arguments passed
    # 解析传递的参数
    
    args = parser.parse_args()

    features = ["mfcc", "chroma", "mel"]
    #  MFCC   
    #  Chromagram
    #  MEL Spectrogram Frequency (mel)
    detector = EmotionRecognizer(estimator_dict[args.model], emotions=args.emotions.split(","), features=features, verbose=0)
    detector.train()
    print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))
    print("Please talk")
    
    filename = "test.wav"
    record_to_file(filename)
    result = detector.predict(filename)
    print(result)
