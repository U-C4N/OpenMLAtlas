# 1. Gerekli kütüphaneleri içe aktarma
# ------------------------------------

# NumPy: Sayılarla (matris, vektör) çalışmamızı sağlayan temel bilimsel hesaplama kütüphanesi.
# 'as np': Artık her yerde 'numpy' yerine kısaca 'np' yazacağız.
import numpy as np

# Matplotlib.pyplot: Grafik çizmemizi sağlar (nokta, doğru, dağılım grafikleri vb.).
# 'plt': Çizim için kullanacağımız kısa isim.
import matplotlib.pyplot as plt

# Pandas: Verileri tablo (Excel sayfası gibi) şeklinde okumamızı ve işlememizi sağlar.
# 'pd': Pandas için standart kısaltma.
import pandas as pd

# Scikit-learn (sklearn): Makine öğrenmesi algoritmalarını barındıran kütüphane.
# LinearRegression: Basit doğrusal regresyon modelini kullanacağız.
from sklearn.linear_model import LinearRegression

# r2_score: Modelimizin ne kadar iyi tahmin yaptığını ölçmek için kullanacağımız skor fonksiyonu.
# (0 ile 1 arasında değer alır, 1'e ne kadar yakınsa o kadar iyi.)
from sklearn.metrics import r2_score

# train_test_split: Veriyi "eğitim" ve "test" olarak ikiye ayıran fonksiyon.
# Böylece modelimizi görmediği veriler üzerinde test edebiliriz.
from sklearn.model_selection import train_test_split


# 2. Veriyi yükleme ve hazır hale getirme
# ---------------------------------------

# pd.read_csv: Aynı klasördeki 'salary.csv' dosyasını okur ve tablo (DataFrame) haline getirir.
# Bu dosyada her satır bir çalışana, sütunlar ise özelliklere (deneyim, maaş) karşılık gelir.
df = pd.read_csv('salary.csv')

# df.head(): İlk 5 satırı ekrana yazar.
# Böylece verinin doğru okunup okunmadığını hızlıca kontrol ederiz.
print(df.head())

# X: Bağımsız değişken(ler) yani giriş özellikleri.
# Burada sadece 'YearsExperience' (deneyim yılı) bilgisini kullanıyoruz.
# Çift köşeli parantez [[...]] kullanmamızın sebebi, sklearn'ün 2 boyutlu bir tablo beklemesidir.
X = df[["YearsExperience"]]

# y: Bağımlı değişken, yani tahmin etmeye çalıştığımız değer.
# Burada çalışanların 'Salary' (maaş) bilgisini tahmin etmek istiyoruz.
y = df["Salary"]



# 3. Veriyi eğitim ve test olarak ayırma
# --------------------------------------

# train_test_split:
# - test_size=0.3: Verinin %30'unu test için ayırırız, %70'i eğitimde kalır.
# - random_state=10: Her çalıştırmada aynı bölünme olsun diye sabit bir sayı veriyoruz.
# X_train, y_train: Modelin "öğrenme" aşamasında göreceği örnekler.
# X_test, y_test: Modelin daha önce görmediği, sadece performansı ölçmek için ayırdığımız örnekler.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# 4. Modeli oluşturma ve eğitme
# -----------------------------

# LinearRegression(): Doğrusal regresyon modelini oluşturur.
model = LinearRegression()

# model.fit: Eğitim (training) adımı.
# Model, X_train -> y_train arasındaki ilişkiyi öğrenir (en uygun doğruyu bulur).
model.fit(X_train, y_train)


# 5. Tahmin yapma
# ---------------

# Eğittiğimiz modeli kullanarak X_test için maaş tahminleri üretiriz.
y_pred = model.predict(X_test)


# 6. Başarıyı ölçme
# -----------------

# r2_score: Gerçek değerler ile (y_test) tahminleri (y_pred) karşılaştırır.
# 1'e yakınsa iyi, 0'a yakınsa zayıf demektir.
r2 = r2_score(y_test, y_pred)
print(f"R2 Skoru: {r2}")

# Üretilen tahminleri görmek için yazdırıyoruz.
print(y_pred)