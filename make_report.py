import pandas as pd
import numpy
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

read_path = os.path.join('./analysis', 'result', 'v1.0', 'fold1_report.csv') 
df = pd.read_csv(read_path)
print(df)
