##from preprocessing import preprocess_image, preprocess_txt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image as keras_image
from metrics import precision, recall, f1

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from numpy import array
from numpy import asarray
from numpy import zeros

import numpy as np


class Classifier:

    def __init__(self, epochs=32, batch_size=64,metrics = False, plot_model_diagram=False, summary=False):
        self.epochs = epochs
        self.metrics = metrics
        self.batch_size = batch_size
        self.plot_model_diagram = plot_model_diagram
        self.summary = summary
        self.seq_len = 50
        # self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = Tokenizer(num_words=10000)
        self.vgg = VGG19(weights='imagenet', include_top=False)
        #self.bert_layer.trainable = False
        self.vgg.trainable = False
        self.embedding_matrix = None

    def encode(self, texts, fit=False):
        if fit:
            self.tokenizer.fit_on_texts(texts)
        texts = self.tokenizer.texts_to_sequences(texts)
        tokenizer = self.tokenizer
        # Adding 1 because of reserved 0 index
        self.vocab_size = len(tokenizer.word_index) + 1
        vocab_size = self.vocab_size

        maxlen = self.seq_len

        texts = pad_sequences(texts, padding='post', maxlen=maxlen)

        if self.embedding_matrix is None:
            embeddings_dictionary = dict()
            glove_file = open('/content/drive/My Drive/colab/data/glove/glove.6B.200d.txt', encoding="utf8")
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = asarray(records[1:], dtype='float32')
                embeddings_dictionary [word] = vector_dimensions
            glove_file.close()

            embedding_matrix = zeros((vocab_size, 200))
            for word, index in tokenizer.word_index.items():
                embedding_vector = embeddings_dictionary.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
            self.embedding_matrix = embedding_matrix

        return texts

    def labelencoder(self, labels):
        new_label = np.zeros((len(labels), 3))
        for i, label in enumerate(labels):
            if label == 0:
                new_label[i] = [1, 0, 0]
            elif label == 1:
                new_label[i] = [0, 1, 0]
            elif label == 2:
                new_label[i] = [0, 0, 1]

        return new_label

    def build(self):
        input_id = Input(shape=(self.seq_len,), dtype=tf.int64,name="text")

        embedding_layer = Embedding(self.vocab_size, 200, weights=[self.embedding_matrix], input_length=self.seq_len , trainable=False)(input_id)
        
        # model.add(LSTM(128,return_sequences=True))
        dense = LSTM(256)(embedding_layer)

        #dense = Dense(768, activation='relu')(bert_out)
        dense = Dense(128, activation='relu')(dense)
        txt_repr = Dropout(0.5)(dense)
        ################################################
        img_in = Input(shape=(224, 224, 3),name="img_imp")
        img_out = self.vgg(img_in)
        flat = Flatten()(img_out)
        dense = Dense(2742, activation='relu')(flat)
        dense = Dense(256, activation='relu')(dense)
        img_repr = Dropout(0.5)(dense)
        concat = Concatenate(axis=1)([img_repr, txt_repr])
        dense = Dense(64, activation='relu')(concat)
        out = Dense(3, activation='softmax')(dense)
        model = Model(inputs=[input_id, img_in], outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(2e-5),
                      metrics=['accuracy', precision, recall, f1]) if self.metrics else model.compile(
            loss='categorical_crossentropy', optimizer=Adam(2e-5), metrics=['accuracy'])

        plot_model(model) if self.plot_model_diagram else None
        model.summary() if self.summary else None

        return model

    def train(self, data, validation_split=0.2):

        input_id =  self.encode(data['text'],fit=True)
        model = self.build()
        image_data = np.asarray(data['image'])
        labels = self.labelencoder(data['label'])

        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        self.history = model.fit([input_id, image_data],
                                 labels,
                                 validation_split=validation_split,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 #callbacks=[es, mc]
                                 )

        model.save_weights('model2')
        self.model_ = model

    def evaluate(self, data):
        model = self.build()
        model.load_weights('model')
        input_id, token_type_id, attention_mask = self.encode(data['text'].apply(preprocess_txt))
        image_data = data['image'].apply(preprocess_image)
        eval_data = [input_id, token_type_id, attention_mask,image_data]
        labels = self.labelencoder(data['label'])
        evaluation = model.evaluate(eval_data, labels)
        return evaluation


    def predict(self, data):

        try:
            # model = self.build()
            # model.load_weights('model2')
            model = self.model_
            input_id = self.encode(data['text'])
            image_data = np.asarray(data['image'])
            labels = self.labelencoder(data['label'])

            value = model.predict([input_id, image_data])

            prediction = np.argmax(value,axis=-1)

            print(*prediction,sep="\n")
            
            return prediction

        except Exception as e:
            print(e)


