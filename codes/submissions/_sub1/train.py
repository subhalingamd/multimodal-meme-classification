import pickle
from model import Classifier

cls = Classifier(epochs=64, batch_size=16, metrics=True, plot_model_diagram=True, summary=True)

with open("data_train.pkl","rb") as pickle_in:
  data = pickle.load(pickle_in)
pickle_in.close()

cls.train(data)

with open("data_test.pkl","rb") as pickle_in:
  data = pickle.load(pickle_in)
pickle_in.close()

cls.predict(data)