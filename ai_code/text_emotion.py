def Text_emotion(text):
    

    from cnsenti import Sentiment#中文文本情感词正负情感词统计
    from cnsenti import Emotion#中文文本情感词正负情感词统计

# senti = Sentiment()
# test_text= '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心'
# result = senti.sentiment_count(test_text)
# print(result)
# Run

# {'words': 24, 
# 'sentences': 2, 
# 'pos': 4, 
# 'neg': 0}

# emotion = Emotion()
# test_text = '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心'
# result = emotion.emotion_count(test_text)
# print(result)
# Run

# {'words': 22, 
# 'sentences': 2, 
# '好': 0, 
# '乐': 4, 
# '哀': 0, 
# '怒': 0, 
# '惧': 0, 
# '恶': 0, 
# '惊': 0}

    senti = Sentiment()
    emotion = Emotion()

    #text = input("请输入语音识别后的文本：")
    pos_neg = senti.sentiment_calculate(text)
    emotion_res = emotion.emotion_count(text)
    pos = pos_neg['pos']
    neg = pos_neg['neg']
    hao = emotion_res['好']
    le = emotion_res['乐']
    ai = emotion_res['哀']
    nu = emotion_res['怒']
    ju = emotion_res['惧']
    wu = emotion_res['恶']
    jing = emotion_res['惊']

    emotion_good = hao * 2 + le * 3 + jing * 1
    emotion_bad = ai * 3 + nu * 2 + wu * 3 + ju * 1 + jing * 1


    score_good = pos * 0.5 + emotion_good * 0.5
    score_bad = neg *0.5 + emotion_bad * 0.5
    

    if score_good > score_bad:
        print('good', score_good)
        txt_mood = 'good'
        txt_score = score_good
    elif score_good < score_bad:
        print('bad', score_bad)
        txt_mood = 'bad'
        txt_score = score_bad
    elif score_good == score_bad:
        print('neutral')
        txt_mood = 'neutral'
        txt_score = score_good

    return txt_mood, txt_score