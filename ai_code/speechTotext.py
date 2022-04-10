
import pyaudio
#print(sr.__version__)

def speechtotext(audio):

    import speech_recognition as sr
    r = sr.Recognizer()
 
    # test = sr.AudioFile('H:\\作业\\专业设计\\大三下\\专业设计\\ai+healthcare\\ai_声音情绪识别\\男声英文词语采样_耳聆网_[声音ID：17345].flac')
    test = sr.AudioFile(audio)
 
    with test as source:
        audio = r.record(source)
 
    type (audio)

    #第一种方法
    print(r.recognize_google(audio, language='zh-CN', show_all= True))#zh-CN
    audio_text = r.recognize_google(audio, language='zh-CN', show_all= True)
    return audio_text
# 简单英文语音识别

# 使用Sphinx识别语音
#第二种方法

# try:
#     print("Sphinx识别结果为:" + r.recognize_sphinx(audio))
# except sr.UnknownValueError:
#     print("无法识别音频")
# except sr.RequestError as e:
#     print("识别错误; {0}".format(e))


# print("调用谷歌在线识别为:", result)
# result = recognizer.recognize_google(audio, language="zh-CN")
