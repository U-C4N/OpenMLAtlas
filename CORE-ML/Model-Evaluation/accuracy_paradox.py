# Accuracy Paradox - Accuracy her zaman guvenilir degildir

# Kutuphaneleri yukluyoruz
import pandas as pd                                     # Veri okumak icin
from sklearn.model_selection import train_test_split    # Veriyi bolmek icin
from sklearn.linear_model import LogisticRegression     # Model icin
from sklearn.preprocessing import StandardScaler        # Veriyi olceklemek icin
from sklearn.metrics import accuracy_score              # Dogruluk hesabi icin

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

# Accuracy hesapla
accuracy = accuracy_score(y_test, y_pred)               # Dogruluk orani

# Sonucu goster
print("=== ACCURACY (DOGRULUK) ===")
print("Accuracy:", accuracy)

# ACCURACY PARADOX NEDIR?
# -----------------------
# Ornek: 1000 kisilik veri
# - 950 kisi saglikli
# - 50 kisi hasta
#
# Aptal model: Herkese "saglikli" der
# - 950 dogru (sagliklilara saglikli dedi)
# - 50 yanlis (hastalara da saglikli dedi)
# - Accuracy = %95 AMA hicbir hastayi bulamadi!
#
# Sonuc: Sadece accuracy'e guvenmek yaniltici olabilir!
