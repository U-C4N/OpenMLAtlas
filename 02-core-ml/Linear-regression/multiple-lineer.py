import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 1. Veriyi Oku
veri = pd.read_csv('veri.csv')

# --- BURASI İSTEDİĞİN KISIM ---

# A) LABEL ENCODING (Cinsiyet sütunu için)
# E ve K harflerini 0 ve 1'e çevirir.
le = LabelEncoder()
veri['Cinsiyet'] = le.fit_transform(veri['Cinsiyet'])

# B) ONE-HOT ENCODING (Sehir sütunu için)
# Şehirleri ayrı sütunlara bölüp 0-1 yapar.
veri = pd.get_dummies(veri, columns=['Sehir'], drop_first=True)

# -----------------------------

# 2. X ve y Belirle
y = veri['Not']
X = veri.drop('Not', axis=1)

# 3. TRAIN ve TEST diye ayır (%20 test olsun)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Modeli Eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Tahmin Yap
tahminler = model.predict(X_test)

# 6. Tabloyu Çiz (En sade hali)
plt.scatter(y_test, tahminler, color='red')
plt.xlabel("Gerçek Notlar")
plt.ylabel("Tahminler")
plt.show()

print("Bitti. Grafik açıldı.")