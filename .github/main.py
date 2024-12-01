from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem


class MyWindget(QMainWindow):
    def __init__(self)
        super().__init__()
        uic.loadUi("u.ui", self)


d1 = pd.read_csv('DATA.csv')
name = d1.institution.to_list()
co = d1.country.to_list()
da = np.log(d1.influence)
d2 = d1.copy()
del d2['institution']
del d2['country']
del d1['publications']
del d1['citations']
del d1['broad_impact']
del d1['year']
del d1['institution']
del d1['country']
d2['broad_impact'].fillna(round(d2['broad_impact'].median()), inplace=True)
d2.influence = da
d3 = d2.copy()
d = d3['world_rank'].to_list()
for i in range(len(d)):
    d[i] = 'Nan'
d3['world_rank'] = d
X = d2.drop('world_rank', axis=1)
Y = d2['world_rank']
lr = LinearRegression()
lr.fit(X, Y)
YN = lr.predict(X)
n = d3['world_rank']
for i in range(len(n)):
    n[i] = round(YN[i])
d3['world_rank'] = n
print('R^2: {0}'.format(lr.score(X, Y)))

if __name__ == "__main__":
    app = QApplication(sys.orig_argv)
    ex = MyWindget()
    ex.show()
    sys.exit(app.exec())
