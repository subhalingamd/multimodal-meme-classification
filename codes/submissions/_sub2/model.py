from preprocessing import preprocess_image, preprocess_txt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Input, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image as keras_image
from metrics import precision, recall, f1

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import numpy as np


class Classifier:

    def __init__(self, epochs=32, batch_size=64,metrics = False, plot_model_diagram=False, summary=False):
        self.epochs = epochs
        self.metrics = metrics
        self.batch_size = batch_size
        self.plot_model_diagram = plot_model_diagram
        self.summary = summary
        self.seq_len = 50
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vgg = VGG19(weights='imagenet', include_top=False)
        self.bert_layer.trainable = False
        self.vgg.trainable = False

    def encode(self, texts):
        input_id = []
        token_type_id = []
        attention_mask = []
        for text in texts:
            dictIn = self.tokenizer.encode_plus(text, max_length=self.seq_len, pad_to_max_length=True, add_special_tokens=True, return_attention_mask = True)
            input_id.append(dictIn['input_ids'])
            token_type_id.append(dictIn['token_type_ids'])
            attention_mask.append(dictIn['attention_mask'])
        return np.array(input_id), np.array(token_type_id), np.array(attention_mask)

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
        input_id = Input(shape=(self.seq_len,), dtype=tf.int64,name="bert_input_id")
        mask_id = Input(shape=(self.seq_len,), dtype=tf.int64,name="bert_mask_id")
        seg_id = Input(shape=(self.seq_len,), dtype=tf.int64,name="bert_seg_id")

        _, bert_out = self.bert_layer([input_id, mask_id, seg_id])
        #bert_out = self.bert_layer([input_id, mask_id, seg_id])[0]
        #dense = GlobalAveragePooling1D()(bert_out)
        dense = Dense(768, activation='relu',kernel_regularizer=regularizers.l2(0.01))(bert_out)
        dense = Dropout(0.2)(dense)
        dense = Dense(128, activation='relu')(dense)
        txt_repr = Dropout(0.5)(dense)
        ################################################
        img_in = Input(shape=(224, 224, 3),name="img_imp")
        img_out = self.vgg(img_in)
        #x= GlobalAveragePooling2D()(x)
        #x= BatchNormalization()(x)
        #x= Dropout(0.5)(x)
        flat = Flatten()(img_out)
        dense = Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)
        dense = Dropout(0.2)(dense)
        dense = Dense(128, activation='relu')(dense)
        img_repr = Dropout(0.4)(dense)
        concat = Concatenate(axis=1)([img_repr, txt_repr])
        dense = Dense(64, activation='relu')(concat)
        dense = Dropout(0.4)(dense)
        out = Dense(3, activation='softmax')(dense)
        model = Model(inputs=[input_id, mask_id, seg_id, img_in], outputs=out)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(2e-5),
                      metrics=['accuracy', precision, recall, f1]) if self.metrics else model.compile(
            loss='categorical_crossentropy', optimizer=Adam(2e-5), metrics=['accuracy'])

        plot_model(model) if self.plot_model_diagram else None
        model.summary() if self.summary else None

        return model

    def train(self, data, validation_split=0.1):

        model = self.build()
        input_id, token_type_id, attention_mask = self.encode(data['text'])
        image_data = np.asarray(data['image'])
        labels = self.labelencoder(data['label'])

        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        self.history = model.fit([input_id, attention_mask, token_type_id, image_data],
                                 labels,
                                 validation_split=validation_split,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 #callbacks=[es, mc]
                                 )

        model.save_weights('model1')

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
            model = self.build()
            model.load_weights('model1')

            input_id, token_type_id, attention_mask = self.encode(data['text'])
            image_data = np.asarray(data['image'])
            labels = self.labelencoder(data['label'])

            value = model.predict([input_id, attention_mask, token_type_id, image_data])

            prediction = np.argmax(value,axis=-1)
            
            print("\n\n")
            print(*prediction,sep="\n")
            return value,prediction

        except Exception as e:
            print(e)

