from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
# # init a model, let's use SVC
my_model = SVC()
# # pass my model to EmotionRecognizer instance
# # and balance the dataset
rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
# # train the model
rec.train()
# # check the test accuracy for that model
# print("Test score:", rec.test_score())
# # check the train accuracy for that model
# print("Train score:", rec.train_score())


# this is a neutral speech from emo-db
print("Prediction:", rec.predict("data/emodb/wav/15a04Nc.wav"))
# this is a sad speech from TESS
print("Prediction:", rec.predict("data/tess_ravdess/validation/Actor_25/25_01_01_01_mob_sad.wav"))