from deep_emotion_recognition import DeepEmotionRecognizer
# initialize instance
# 初始化实例
# inherited from emotion_recognition.EmotionRecognizer 继承于
# default parameters (LSTM: 128x2, Dense:128x2)    **默认的参数
deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], 
n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
# 训练模型
deeprec.train()
# get the accuracy
# 打印精确度
print(deeprec.test_score())
# predict angry audio sample
# 预测  ”生气“ 的例子
prediction = deeprec.predict('data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav')
print(f"Prediction: {prediction}")
print("******Predicting probabilities is also possible (for classification ofc):*****")
print(deeprec.predict_proba("data/emodb/wav/16a01Wb.wav"))
print("********Confusion Matrix*********")
print(deeprec.confusion_matrix(percentage=True, labeled=True))
