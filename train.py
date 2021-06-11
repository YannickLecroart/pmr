from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday, Easter
from pandas.tseries.offsets import Day, CustomBusinessDay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


df = pd.read_excel("parking_data_full.xlsx", index_col=0)

df.index = pd.to_datetime(df.index)

df = df[~df.index.duplicated()]

df = df.asfreq(('15min'))

df = df.fillna(method='ffill')


class FrenchJoursFeries(AbstractHolidayCalendar):
    """ Custom Holiday calendar for France based on
        https://en.wikipedia.org/wiki/Public_holidays_in_France
      - 1 January: New Year's Day
      - Moveable: Easter Monday (Monday after Easter Sunday)
      - 1 May: Labour Day
      - 8 May: Victory in Europe Day
      - Moveable Ascension Day (Thursday, 39 days after Easter Sunday)
      - 14 July: Bastille Day
      - 15 August: Assumption of Mary to Heaven
      - 1 November: All Saints' Day
      - 11 November: Armistice Day
      - 25 December: Christmas Day
    """
    rules = [
        Holiday('New Years Day', month=1, day=1),
        EasterMonday,
        Holiday('Labour Day', month=5, day=1),
        Holiday('Victory in Europe Day', month=5, day=8),
        Holiday('Ascension Day', month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday('Bastille Day', month=7, day=14),
        Holiday('Assumption of Mary to Heaven', month=8, day=15),
        Holiday('All Saints Day', month=11, day=1),
        Holiday('Armistice Day', month=11, day=11),
        Holiday('Christmas Day', month=12, day=25)
    ]

cal = FrenchJoursFeries()

holidays = cal.holidays(start=df.index.min(), end=df.index.max())



def create_features(df, label=None):#input/output features pour algo
        """
        Creates time series features from datetime index
        """
        df['date'] = df.index
        df['heure'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['jour_semaine'] = df['date'].dt.dayofweek
        df['trimestre'] = df['date'].dt.quarter
        df['mois'] = df['date'].dt.month
        df['annee'] = df['date'].dt.year
        df['jour_annee'] = df['date'].dt.dayofyear
        df['jour_mois'] = df['date'].dt.day
        df['semaine'] = df['date'].dt.week
        seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
        month_to_season = dict(zip(range(1,13), seasons))
        df['saison'] = df['date'].dt.month.map(month_to_season)
        df['jour_ferie'] = df.index.isin(holidays)
        df['jour_ferie'] = df['jour_ferie'].astype(int)
        bins = [0,6,11,13,19,23]#0-6h nuit/7-12h matin/12-14h pause dejeuner/14-20h apres midi/20-24h soir 
        labels = [0,1,2,3,4]#nuit/matin/pausedejeuner/apresmidi/soir
        df['plage_horaire'] = pd.cut(df['heure'], bins=bins, labels=labels, include_lowest=True)#to include 0
        df['plage_horaire'] = df['plage_horaire'].astype(int)

        df = df.loc[~((df['heure'] < 8) | (df['heure'] > 19))]


        X = df[['heure', 'jour_semaine', 'minute','trimestre','mois','annee','jour_annee', 'jour_mois', 'semaine','saison', 'jour_ferie', 'plage_horaire']]
        if label:
            y = df[label]
            return X, y
        return X


X, y = create_features(df, label='presence') 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators = 1000, max_depth=10, verbose=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
    
accuracy = accuracy_score(y_test, y_pred)
accuracy = round(accuracy * 100, 1)
    
with open('accuracy_score.txt', 'w') as f:
    f.write(str(accuracy))

# Confusion Matrix and plot
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig("cm.png",dpi=120) 
plt.close()
# Print classification report
cr = classification_report(y_test, model.predict(X_test))
with open('classification_report.txt', 'w') as f:
    f.write(str(cr))

# Plot the ROC curve
model_ROC = plot_roc_curve(model, X_test, y_test)
plt.tight_layout()
plt.savefig("roc.png",dpi=120) 
plt.close()