import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import random
import pandas as pd
from tabpfn import TabPFNClassifier

model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
def train():
  done = pd.read_csv('./demand.csv',index_col=False)
  X = np.array(done[["doys","dows"]])
  y = list(done["pers"])
  X,y = shuffle(X, y, random_state=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
  model.fit(X_train, y_train)
  y_eval, p_eval = model.predict(X_test, return_winning_probability=True)
  print('Accuracy', accuracy_score(y_test, y_eval))
  y_eval, p_eval = model.predict(X_train, return_winning_probability=True)
  print('Accuracy', accuracy_score(y_train, y_eval))
  return model

train()
def mess_pred(doy,dow):
    predictions = model.predict([[doy,dow]])
    return predictions*1487/5

def get_mess_data():
    data = pd.DataFrame({
        'Meal': ['Breakfast', 'Lunch', 'Dinner'],
        'A_Footfall': [int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10), int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10), int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10)],
        'C_Footfall': [int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10), int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10), int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10)],
        'D_Footfall': [int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10), int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10), int((mess_pred(random.randint(0,366),random.randint(0,7))[0])*random.randint(1,10)/10)]
    })
    return data