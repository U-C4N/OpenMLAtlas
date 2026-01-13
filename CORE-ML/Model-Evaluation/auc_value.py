# AUC Value - ROC egrisinin altindaki alan

# Kutuphaneleri yukluyoruz
import pandas as pd                                     # Veri okumak icin
import matplotlib.pyplot as plt                         # Grafik cizmek icin
from sklearn.model_selection import train_test_split    # Veriyi bolmek icin
from sklearn.linear_model import LogisticRegression     # Model icin
from sklearn.preprocessing import StandardScaler        # Veriyi olceklemek icin
from sklearn.preprocessing import LabelEncoder          # Metni sayiya cevirmek icin
from sklearn.metrics import roc_curve                   # ROC egrisi icin
from sklearn.metrics import roc_auc_score               # AUC degeri icin

# Veriyi oku
df = pd.read_csv('data.csv')                            # CSV dosyasini oku

# Ozellikleri ve hedefi ayir
X = df.drop(['Country', 'Gender'], axis=1)              # Ozellikler
y = df['Gender']                                        # Hedef

# Gender'i sayiya cevir
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

# Olasilik tahmini al
y_proba = model.predict_proba(X_test)[:, 1]             # Pozitif sinif olasiligi

# AUC degerini hesapla
auc = roc_auc_score(y_test, y_proba)                    # AUC skoru

# ROC egrisi icin degerler
fpr, tpr, thresholds = roc_curve(y_test, y_proba)       # FPR ve TPR

# Grafik ciz
plt.plot(fpr, tpr, label='ROC Egrisi')                  # Mavi cizgi
plt.plot([0, 1], [0, 1], color='red', linestyle='--')   # Kirmizi kesikli cizgi
plt.fill_between(fpr, tpr, alpha=0.3)                   # Alani boya (acik mavi)
plt.xlabel('False Positive Rate')                       # X ekseni etiketi
plt.ylabel('True Positive Rate')                        # Y ekseni etiketi
plt.title('AUC = ' + str(round(auc, 2)))                # Baslikta AUC degerini goster
plt.show()                                              # Grafigi goster

# Sonucu goster
print("=== AUC DEGERI ===")
print("AUC:", round(auc, 2))

# AUC NEDIR?
# ----------
# AUC = Grafikteki mavi alanin buyuklugu
#
# AUC = 1.0 -> Mukemmel (tum alan mavi)
# AUC = 0.5 -> Rastgele tahmin (kotu)
# AUC = 0.0 -> Tamamen yanlis (modeli ters cevir)
#
# Bizim modelimiz ne kadar iyi?
# AUC 0.7'den buyukse fena degil
# AUC 0.8'den buyukse iyi
# AUC 0.9'dan buyukse cok iyi
