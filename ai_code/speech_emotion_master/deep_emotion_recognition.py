import os
# disable keras loggings
# 静止使用 keras 的日志记录功能
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
#from tensorflow.keras import layers

from tensorflow.python.keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout 

# GRU 门循环单元

from tensorflow.python.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.utils.np_utils import to_categorical



from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

from data_extractor import load_data
from create_csv import write_custom_csv, write_emodb_csv, write_tess_ravdess_csv
from emotion_recognition import EmotionRecognizer
from utils import get_first_letters, AVAILABLE_EMOTIONS, extract_feature, get_dropout_str

import numpy as np
import pandas as pd
import random


class DeepEmotionRecognizer(EmotionRecognizer):
    """
    The Deep Learning version of the Emotion Recognizer.
    This class uses RNN (LSTM, GRU, etc.) and Dense layers.
    #TODO add CNNs
    """
    # 这个深度学习框架使用了 RNN（LSTM，GRU ，etc） 
    def __init__(self, **kwargs):
        """
        params:
            emotions (list): list of emotions to be used. Note that these emotions must be available in
                RAVDESS_TESS & EMODB Datasets, available nine emotions are the following:
                    'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps' ( pleasant surprised ), 'boredom' （令人厌倦的）.
                Default is ["sad", "neutral", "happy"].
            tess_ravdess (bool): whether to use TESS & RAVDESS Speech datasets, default is True. 

            是否使用语音识别数据集，默认是使用的：

            emodb (bool): whether to use EMO-DB Speech dataset, default is True.
            custom_db (bool): whether to use custom Speech dataset that is located in `data/train-custom`
                and `data/test-custom`, default is True.


            tess_ravdess_name (str): the name of the output CSV file for TESS&RAVDESS dataset, default is "tess_ravdess.csv".
            emodb_name (str): the name of the output CSV file for EMO-DB dataset, default is "emodb.csv".
            custom_db_name (str): the name of the output CSV file for the custom dataset, default is "custom.csv".
            features (list): list of speech features to use, default is ["mfcc", "chroma", "mel"]
                (i.e MFCC, Chroma and MEL spectrogram ).

            classification (bool): whether to use classification or regression, default is True.
            使用分类和回归

            balance (bool): whether to balance the dataset ( both training and testing ), default is True.
            平衡数据
            verbose (bool/int): whether to print messages on certain tasks.
            打印任务的消息
            ==========================================================


            Model params
            n_rnn_layers (int): number of RNN layers, default is 2.
            RNN 的层数
            cell (keras.layers.RNN instance): RNN cell used to train the model, default is LSTM.
            使用 RNN 单元去训练模型
            rnn_units (int): number of units of `cell`, default is 128.
            单元格默认的单位数是 128
            n_dense_layers (int): number of Dense layers, default is 2.
            # 默认的 Dense 是两层
            dense_units (int): number of units of the Dense layers, default is 128.
            # 默认是 128 个单元格
            dropout (list/float): dropout rate,  （drpout的参数）
                - if list,  it indicates the dropout rate of  each layer.
                - if float, it indicates the dropout rate for all layers.
                Default is 0.3.
            ==========================================================

            Training params
            batch_size (int): number of samples per gradient update, default is 64.
            batch 的数据集批次
            epochs (int): number of epochs, default is 1000.
            轮询次数

            optimizer (str/keras.optimizers.Optimizer instance): optimizer used to train, default is "adam".
            优化器默认使用的是 adam

            loss (str/callback from keras.losses): loss function that is used to minimize during training,
                default is "categorical_crossentropy" for classification and "mean_squared_error" for 
                regression.
            损失函数：categorical_crossentropy

        """
        # init EmotionRecognizer
        # 载入 情绪
        super().__init__(None, **kwargs)

        self.n_rnn_layers = kwargs.get("n_rnn_layers", 2)
        self.n_dense_layers = kwargs.get("n_dense_layers", 2)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.dense_units = kwargs.get("dense_units", 128)
        self.cell = kwargs.get("cell", LSTM)

        # list of dropouts of each layer
        # must be len(dropouts) = n_rnn_layers + n_dense_layers
        # 设置 dropouts 相关的参数
        self.dropout = kwargs.get("dropout", 0.3)
        self.dropout = self.dropout if isinstance(self.dropout, list) else [self.dropout] * ( self.n_rnn_layers + self.n_dense_layers )
        # number of classes ( emotions )
        # 情绪的分类数目
        self.output_dim = len(self.emotions)

        # optimization attributes
        # 优化属性
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "categorical_crossentropy")

        # training attributes
        # 训练特性
        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 1000)
        
        # the name of the model
        # 模型的名字
        self.model_name = ""
        self._update_model_name()

        # init the model
        # 初始化模型
        self.model = None

        # compute the input length
        # 计算输入长度
        self._compute_input_length()

        # boolean attributes
        # 布尔系特性
        self.model_created = False

    def _update_model_name(self):
        """
        Generates a unique model name based on parameters passed and put it on `self.model_name`.

        *************************************************
        基于参数，生成唯一的模型名字，并把他存入 self.model_name
        *************************************************

        This is used when saving the model.
        # 这个是在保存模型的时候使用。
        """


        # get first letters of emotions, for instance: 获得情绪的首字母

        # ["sad", "neutral", "happy"] => 'HNS' (sorted alphabetically)
        # 按照字母顺序排序
        emotions_str = get_first_letters(self.emotions)
        # 'c' for classification & 'r' for regression
        # c 表示分类， R 表示回归.
        problem_type = 'c' if self.classification else 'r'
        dropout_str = get_dropout_str(self.dropout, n_layers=self.n_dense_layers + self.n_rnn_layers)
        self.model_name = f"{emotions_str}-{problem_type}-{self.cell.__name__}-layers-{self.n_rnn_layers}-{self.n_dense_layers}-units-{self.rnn_units}-{self.dense_units}-dropout-{dropout_str}.h5"

    def _get_model_filename(self):
        """Returns the relative path of this model name"""
        # 返回相对路径   绝对路径是(absolute path)
        return f"results/{self.model_name}"

    def _model_exists(self):
        """
        Checks if model already exists in disk, returns the filename,
        and returns `None` otherwise.
        """
        # 检测模型是否已经存在，如果存在的话，返回模型的文件名。否则返回 None

        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None

    def _compute_input_length(self):
        """
        Calculates the input shape to be able to construct the model.
        """ 
        # 计算输入模型，以方便构造模型.

        if not self.data_loaded:
            self.load_data()
        self.input_length = self.X_train[0].shape[1]

    def _verify_emotions(self):
        super()._verify_emotions()
        self.int2emotions = {i: e for i, e in enumerate(self.emotions)}
        self.emotions2int = {v: k for k, v in self.int2emotions.items()}

    def create_model(self):
        """
        Constructs the neural network based on parameters passed.
        """
        # 根据所传递的参数构造神经网络。
        if self.model_created:
            # model already created, why call twice
            # 模型已经存在了，为什么要 Call 两次
            return

        if not self.data_loaded:
            # if data isn't loaded yet, load it
            # 已经存在，那么就加载他。
            self.load_data()
        
        model = Sequential()

        # rnn layers
        # RNN 层
        for i in range(self.n_rnn_layers):
            if i == 0:
                # first layer
                model.add(self.cell(self.rnn_units, return_sequences=True, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i]))
            else:
                # middle layers
                model.add(self.cell(self.rnn_units, return_sequences=True))
                model.add(Dropout(self.dropout[i]))

        if self.n_rnn_layers == 0:
            i = 0

        # dense layers
        # dense 层
        for j in range(self.n_dense_layers):
            # if n_rnn_layers = 0, only dense
            # 如果 RNN 层的层数=0 ，的话，那么就只有 dense 层
            if self.n_rnn_layers == 0 and j == 0:
                model.add(Dense(self.dense_units, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i+j]))
            else:
                model.add(Dense(self.dense_units))
                model.add(Dropout(self.dropout[i+j]))
                
        if self.classification:
            model.add(Dense(self.output_dim, activation="softmax"))
            model.compile(loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer)
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], optimizer=self.optimizer)
        
        self.model = model
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def load_data(self):
        """
        Loads and extracts features from the audio files for the db's specified.
        And then reshapes the data.
        """
        # 从 db 中加载并抽取特征
        # 重塑这些个数据

        super().load_data()
        # reshape X's to 3 dims
        # 重新塑形
        X_train_shape = self.X_train.shape
        X_test_shape = self.X_test.shape
        self.X_train = self.X_train.reshape((1, X_train_shape[0], X_train_shape[1]))
        self.X_test = self.X_test.reshape((1, X_test_shape[0], X_test_shape[1]))

        if self.classification:
            # one-hot encode when its classification
            # 分类的时候，使用  one—hot 的方法
            self.y_train = to_categorical([ self.emotions2int[str(e)] for e in self.y_train ])
            self.y_test = to_categorical([ self.emotions2int[str(e)] for e in self.y_test ])
        
        # reshape labels
        # 重塑标签
        y_train_shape = self.y_train.shape
        y_test_shape = self.y_test.shape
        if self.classification:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], y_train_shape[1]))    
            self.y_test = self.y_test.reshape((1, y_test_shape[0], y_test_shape[1]))
        else:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], 1))
            self.y_test = self.y_test.reshape((1, y_test_shape[0], 1))

    def train(self, override=False):
        """
        Trains the neural network.
        Params:
            override (bool): whether to override the previous identical model, can be used
                when you changed the dataset, default is False
        """
        # 训练神经网络
        # 是否覆盖掉你之前的模型，他是在你改变数据集的时候，默认是关闭的。。


        # if model isn't created yet, create it
        # 如果模型还没有创建他的话，创建它。
        if not self.model_created:
            self.create_model()



        '''
          if the model already exists and trained, just load the weights and return
          but if override is True, then just skip loading weights
         '''
        # 如果模型已经存在了，或者是训练过了，那么加载模型就可以了。
        # 如果override 是 true ，那么跳过加载权重。
        # but if override is True, then just skip loading weights
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))

        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[self.checkpointer, self.tensorboard],
                        verbose=self.verbose)
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, audio_path):
        feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
        if self.classification:
            return self.int2emotions[self.model.predict_classes(feature)[0][0]]
        else:
            return self.model.predict(feature)[0][0][0]

    def predict_proba(self, audio_path):
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
            proba = self.model.predict(feature)[0][0]
            result = {}
            for prob, emotion in zip(proba, self.emotions):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Probability prediction doesn't make sense for regression")



    def test_score(self):
        y_test = self.y_test[0]
        if self.classification:
            y_pred = self.model.predict_classes(self.X_test)[0]
            y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
            return accuracy_score(y_true=y_test, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_test)[0]
            return mean_absolute_error(y_true=y_test, y_pred=y_pred)

    def train_score(self):
        y_train = self.y_train[0]
        if self.classification:
            y_pred = self.model.predict_classes(self.X_train)[0]
            y_train = [np.argmax(y, out=None, axis=None) for y in y_train]
            return accuracy_score(y_true=y_train, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_train)[0]
            return mean_absolute_error(y_true=y_train, y_pred=y_pred)

    def confusion_matrix(self, percentage=True, labeled=True):
        """Compute confusion matrix to evaluate the test accuracy of the classification"""
        # 绘制混淆举证，测试分类的精度
        if not self.classification:
            raise NotImplementedError("Confusion matrix works only when it is a classification problem")
        y_pred = self.model.predict_classes(self.X_test)[0]
        # invert from keras.utils.to_categorical
        # 从 keras  里面转化。。。
        y_test = np.array([ np.argmax(y, axis=None, out=None) for y in self.y_test[0] ])
        matrix = confusion_matrix(y_test, y_pred, labels=[self.emotions2int[e] for e in self.emotions]).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            # 百分比化显示.
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[ f"true_{e}" for e in self.emotions ],
                                    columns=[ f"predicted_{e}" for e in self.emotions ])
        return matrix

    def n_emotions(self, emotion, partition):
        """Returns number of `emotion` data samples in a particular `partition`
        ('test' or 'train')
        """
        # 从特定的部分返回情绪数据例子的数量 (”训练“ 和  ”测试“)



        if partition == "test":
            if self.classification:
                y_test = np.array([ np.argmax(y, axis=None, out=None)+1 for y in np.squeeze(self.y_test) ]) 
            else:
                y_test = np.squeeze(self.y_test)
            return len([y for y in y_test if y == emotion])
        elif partition == "train":
            if self.classification:
                y_train = np.array([ np.argmax(y, axis=None, out=None)+1 for y in np.squeeze(self.y_train) ])
            else:
                y_train = np.squeeze(self.y_train)
            return len([y for y in y_train if y == emotion])

    def get_samples_by_class(self):
        """
        Returns a dataframe that contains the number of training 
        and testing samples for all emotions
        """
        # 返回文本框。包含所有的测试和训练数据。
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.emotions:
            n_train = self.n_emotions(self.emotions2int[emotion]+1, "train")
            n_test = self.n_emotions(self.emotions2int[emotion]+1, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)
        
        # get total
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(data={"train": train_samples, "test": test_samples, "total": total}, index=self.emotions + ["total"])

    def get_random_emotion(self, emotion, partition="train"):
        """
        Returns random `emotion` data sample index on `partition`
        """
        # 返回随机的样本索引，基于划分区的。
        if partition == "train":
            y_train = self.y_train[0]
            index = random.choice(list(range(len(y_train))))
            element = self.int2emotions[np.argmax(y_train[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_train))))
                element = self.int2emotions[np.argmax(y_train[index])]
        elif partition == "test":
            y_test = self.y_test[0]
            index = random.choice(list(range(len(y_test))))
            element = self.int2emotions[np.argmax(y_test[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_test))))
                element = self.int2emotions[np.argmax(y_test[index])]
        else:
            raise TypeError("Unknown partition, only 'train' or 'test' is accepted")

        return index

    def determine_best_model(self, train=True):
        # TODO
        raise TypeError("This method isn't supported yet for deep nn")


if __name__ == "__main__":
    rec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'],
                                epochs=300, verbose=0)
    rec.train(override=False)
    print("Test accuracy score:", rec.test_score() * 100, "%")