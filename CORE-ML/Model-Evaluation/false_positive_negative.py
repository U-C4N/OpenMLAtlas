# False Positive ve False Negative - Iki tur yanlis tahmin

# Kutuphaneleri yukluyoruz
import pandas as pd                                     # Veri okumak icin
from sklearn.model_selection import train_test_split    # Veriyi bolmek icin
from sklearn.linear_model import LogisticRegression     # Model icin
from sklearn.preprocessing import StandardScaler        # Veriyi olceklemek icin
from sklearn.metrics import confusion_matrix            # Sonuc tablosu icin

# Veriyi oku
df = pd.read_csv('data.csv')                            # CSV dosyasini oku

# Ozellikleri ve hedefi ayir
X = df.drop(['Country', 'Gender'], axis=1)              # Ozellikler
y = df['Gender']                                        # Hedef

# Veriyi bol
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Verileri olcekle
scaler = StandardScaler()                               # Olcekleyici olustur
X_train = scaler.fit_transform(X_train)                 # Egitim verisini olcekle
X_test = scaler.transform(X_test)                       # Test verisini olcekle

# Modeli olustur ve egit
model = LogisticRegression(random_state=0)              # Model olustur
model.fit(X_train, y_train)                             # Modeli egit

# Tahmin yap
y_pred = model.predict(X_test)                          # Tahminleri al

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)                   # Tabloyu olustur

# Degerleri ayir
TN = cm[0][0]                                           # True Negative
FP = cm[0][1]                                           # False Positive
FN = cm[1][0]                                           # False Negative
TP = cm[1][1]                                           # True Positive

# Sonuclari goster
print("=== FALSE POSITIVE (Tip 1 Hata) ===")
print("Sayisi:", FP)
print("f oldugu halde m tahmin ettik")
print()

print("=== FALSE NEGATIVE (Tip 2 Hata) ===")
print("Sayisi:", FN)
print("m oldugu halde f tahmin ettik")

# FALSE POSITIVE ve FALSE NEGATIVE NEDIR?
# ---------------------------------------
# Ornek: Hastalik testi yapiyoruz
#
# FALSE POSITIVE (Yanlis Pozitif):
# - Test "Hastasin" dedi ama aslinda sagliklisin
# - Yanlis alarm! Gereksiz panik yaptirdik
#
# FALSE NEGATIVE (Yanlis Negatif):
# - Test "Sagliklisin" dedi ama aslinda hastasin
# - Hastaligi kacirdik! Tehlikeli durum
#
# Hangisi daha kotu?
# - Kanser testi: FN daha kotu (hastayi kaciririz)
# - Spam filtresi: FP daha kotu (onemli mail kaybolur)
