# Confusion Matrix - Modelin dogru ve yanlis tahminlerini gosteren tablo

# Kutuphaneleri yukluyoruz
import pandas as pd                                     # Veri okumak icin
from sklearn.model_selection import train_test_split    # Veriyi bolmek icin
from sklearn.linear_model import LogisticRegression     # Model icin
from sklearn.preprocessing import StandardScaler        # Veriyi olceklemek icin
from sklearn.metrics import confusion_matrix            # Sonuc tablosu icin

# Veriyi oku
df = pd.read_csv('data.csv')                            # CSV dosyasini oku

# X = Tahmin icin kullanilacak ozellikler (boy, kilo, yas)
# y = Tahmin edilecek hedef (cinsiyet)
X = df.drop(['Country', 'Gender'], axis=1)              # Country ve Gender'i cikar
y = df['Gender']                                        # Gender'i hedef olarak al

# Veriyi egitim ve test olarak bol
# %67 egitim, %33 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Verileri olcekle (buyuk-kucuk sayilari dengele)
scaler = StandardScaler()                               # Olcekleyici olustur
X_train = scaler.fit_transform(X_train)                 # Egitim verisini olcekle
X_test = scaler.transform(X_test)                       # Test verisini olcekle

# Modeli olustur ve egit
model = LogisticRegression(random_state=0)              # Model olustur
model.fit(X_train, y_train)                             # Modeli egit

# Test verisiyle tahmin yap
y_pred = model.predict(X_test)                          # Tahminleri al

# Confusion Matrix olustur
cm = confusion_matrix(y_test, y_pred)                   # Gercek vs Tahmin tablosu

# Sonuclari goster
print("=== CONFUSION MATRIX ===")
print(cm)
print()
print("TN:", cm[0][0], "- Dogru f tahmini")             # True Negative
print("FP:", cm[0][1], "- Yanlis m tahmini")            # False Positive
print("FN:", cm[1][0], "- Yanlis f tahmini")            # False Negative
print("TP:", cm[1][1], "- Dogru m tahmini")             # True Positive

# CONFUSION MATRIX NEDIR?
# -----------------------
# TN = f dedik, gercekten f. Dogru!
# TP = m dedik, gercekten m. Dogru!
# FP = m dedik ama aslinda f. Yanlis!
# FN = f dedik ama aslinda m. Yanlis!
