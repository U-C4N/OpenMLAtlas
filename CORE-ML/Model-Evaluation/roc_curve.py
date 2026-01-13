# ROC Curve - Modelin performansini gosteren egri

# Kutuphaneleri yukluyoruz
import pandas as pd                                     # Veri okumak icin
import matplotlib.pyplot as plt                         # Grafik cizmek icin
from sklearn.model_selection import train_test_split    # Veriyi bolmek icin
from sklearn.linear_model import LogisticRegression     # Model icin
from sklearn.preprocessing import StandardScaler        # Veriyi olceklemek icin
from sklearn.preprocessing import LabelEncoder          # Metni sayiya cevirmek icin
from sklearn.metrics import roc_curve                   # ROC egrisi icin

# Veriyi oku
df = pd.read_csv('data.csv')                            # CSV dosyasini oku

# Ozellikleri ve hedefi ayir
X = df.drop(['Country', 'Gender'], axis=1)              # Ozellikler
y = df['Gender']                                        # Hedef

# Gender'i sayiya cevir (ROC icin gerekli)
le = LabelEncoder()                                     # Encoder olustur
y = le.fit_transform(y)                                 # f=0, m=1 olarak cevir

# Veriyi bol
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Verileri olcekle
scaler = StandardScaler()                               # Olcekleyici olustur
X_train = scaler.fit_transform(X_train)                 # Egitim verisini olcekle
X_test = scaler.transform(X_test)                       # Test verisini olcekle

# Modeli olustur ve egit
model = LogisticRegression(random_state=0)              # Model olustur
model.fit(X_train, y_train)                             # Modeli egit

# Olasilik tahmini al (0-1 arasi deger)
y_proba = model.predict_proba(X_test)[:, 1]             # Pozitif sinif olasiligi

# ROC degerlerini hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_proba)       # FPR, TPR ve esik degerleri

# ROC egrisini ciz
plt.plot(fpr, tpr)                                      # Mavi cizgi (ROC)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')   # Kirmizi kesikli cizgi
plt.xlabel('False Positive Rate')                       # X ekseni etiketi
plt.ylabel('True Positive Rate')                        # Y ekseni etiketi
plt.title('ROC Egrisi')                                 # Baslik
plt.show()                                              # Grafigi goster

# ROC EGRISI NEDIR?
# -----------------
# Mavi cizgi = Bizim modelimiz
# Kirmizi cizgi = Rastgele tahmin (yazi-tura gibi)
#
# Mavi cizgi sol ust koseye ne kadar yakinsa model o kadar iyi
# Mavi cizgi kirmizi cizginin ustundeyse model ise yariyor demektir
