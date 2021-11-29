import pickle
from model2 import Classifier

cls = Classifier(epochs=48, batch_size=32, metrics=True, plot_model_diagram=True, summary=True)

with open("data_train.pkl","rb") as pickle_in:
  train_data = pickle.load(pickle_in)
pickle_in.close()

#cls.train(data)


with open("data_test.pkl","rb") as pickle_in:
  test_data = pickle.load(pickle_in)
pickle_in.close()

cls.train_and_predict(train_data,test_data)
