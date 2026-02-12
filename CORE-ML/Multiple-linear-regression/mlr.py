# Çoklu Doğrusal Regresyon Örneği
# --------------------------------
# Bu dosyada bir öğrencinin notunu (Grade), birden fazla özelliğe bakarak
# tahmin etmeye çalışıyoruz. Örneğin: cinsiyet, yaş, şehir vb.

# Gerekli kütüphaneler
import pandas as pd                 # Veri okuma ve tablo işlemleri
import matplotlib.pyplot as plt      # Sonuçları görselleştirmek için
from sklearn.linear_model import LinearRegression  # Çoklu doğrusal regresyon modeli
from sklearn.model_selection import train_test_split  # Veriyi eğitim/test olarak bölmek için
from sklearn.preprocessing import LabelEncoder        # Kategorik değişkenleri sayıya çevirmek için
from sklearn.metrics import r2_score                  # Model başarısını ölçmek için

# 1. Veriyi okuma
# ---------------
# Aynı klasördeki data.csv dosyasını okuyoruz.
data = pd.read_csv('data.csv')

# 2. Kategorik veriyi sayıya çevirme
# -----------------------------------
# 'Gender' sütunu metin ("Male", "Female" gibi). Makine öğrenmesi modelleri doğrudan
# metinle çalışamadığı için önce bunları sayıya çeviriyoruz.
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Şehir bilgisini (City) de one-hot encoding ile sayısal hale getiriyoruz.
# get_dummies: Her şehir için ayrı bir sütun oluşturur (0 veya 1 değer alır).
# drop_first=True: İlk şehri atarak gereksiz fazladan sütun oluşmasını engeller.
data = pd.get_dummies(data, columns=['City'], drop_first=True)

# 3. Özellikler (X) ve hedef değişken (y)
# --------------------------------------
# Grade: Tahmin etmeye çalıştığımız hedef değişken (öğrencinin notu).
# Diğer tüm sütunlar: Modelin kullanacağı giriş özellikleri.
x = data.drop('Grade', axis=1)  # Özellikler (giriş)
y = data['Grade']               # Hedef (çıktı)

# 4. Veriyi eğitim ve test olarak ayırma
# --------------------------------------
# test_size=0.2: Verinin %20'si test için ayrılır, %80'i eğitimde kullanılır.
# random_state=42: Her seferinde aynı bölünmeyi elde etmek için sabit sayı.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Modeli oluşturma ve eğitme
# -----------------------------
model = LinearRegression()
model.fit(x_train, y_train)

# 6. Test verisi üzerinde tahmin yapma
# ------------------------------------
y_pred = model.predict(x_test)

# 7. Gerçek ve tahmin edilen notları yan yana gösterme
# ----------------------------------------------------
sonuc = pd.DataFrame({
    'Actual_Grade': y_test.values,     # Gerçek notlar
    'Predicted_Grade': y_pred          # Modelin tahmin ettiği notlar
})

print(sonuc)

# 8. Model başarısını ölçme (R^2 skoru)
# -------------------------------------
r2 = r2_score(y_test, y_pred)
print("R^2 skoru:", r2)

# 9. Sonuçları görselleştirme
# ---------------------------
# X ekseni: Gerçek notlar
# Y ekseni: Tahmin edilen notlar
# Noktalar doğruya ne kadar yakınsa, model o kadar iyidir.
plt.scatter(y_test, y_pred, color='red')
plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.title('Actual vs Predicted Grade')
plt.show()
