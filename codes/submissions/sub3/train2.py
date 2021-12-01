import pickle
from model2 import Classifier

cls = Classifier(epochs=32, batch_size=32, metrics=True, plot_model_diagram=True, summary=True)

with open("data_train1.pkl","rb") as pickle_in:
  data = pickle.load(pickle_in)
pickle_in.close()

cls.train(data)


with open("data_test1.pkl","rb") as pickle_in:
  data = pickle.load(pickle_in)
pickle_in.close()

cls.predict(data)
