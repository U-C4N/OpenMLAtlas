import pandas as pd                      # pandas, veri işleme ve analiz için kullanılır
import statsmodels.api as sm             # statsmodels, istatistiksel modellemeler ve OLS için gerekli
from sklearn.model_selection import train_test_split  # Veri kümesini eğitim/teste ayırmak için gerekli
from sklearn.preprocessing import LabelEncoder        # Kategorik verileri sayısala çevirmek için

# Veri setini okuyoruz. 'data.csv' içindeki verileri bir DataFrame'e aktarıyoruz.
data = pd.read_csv('data.csv')

# 'Gender' sütunu kategorik olduğu için (örn: Male/Female), sayısal formata çevirmek gerekir.
# LabelEncoder bunu 0 ve 1 gibi değerlerle değiştirir.
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# 'City' sütunu da kategorik olduğu için one-hot encoding (dummy değişkenler) yapıyoruz.
# drop_first=True ile ilk kategoriyi atıp, sahte korelasyonları (dummy variable trap) önlüyoruz.
# dtype=int, dummy sütunların tipini int yapar.
data = pd.get_dummies(data, columns=['City'], drop_first=True, dtype=int)

# Bağımsız değişkenleri (özellikler) ve bağımlı değişkeni (hedef) ayırıyoruz.
# Bağımlı değişken (Grade) modelden çıkarılıyor. Model bunu tahmin etmeye çalışacak.
x = data.drop('Grade', axis=1)
y = data['Grade']

# OLS modelinde sabit terimi (bias/intercept) de modelleyebilmek için 'const' sütununu tüm x'e ekliyoruz.
# Bunu train_test_split'ten önce yapmamız önemli, çünkü ayrı ayrı eklersek eğitim ve testde tutarsızlık olabilir.
# Yani, önce tüm veri setinde aynı yapıyı oluşturur, sonra bölme işlemi yaparız.
x = sm.add_constant(x)

# Veri kümesini eğitim ve test olarak %80-%20 oranında rastgele ayırıyoruz.
# Böylece eğitim verisiyle modeli kurup, test verisiyle performansı kontrol edebiliriz.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# OLS (Ordinary Least Squares) modelini yalnızca eğitim verisiyle kuruyoruz.
model = sm.OLS(y_train, x_train)

# Modeli veriye uyduruyoruz (fit). Burada regresyon katsayıları öğreniliyor.
results = model.fit()

# Modelin özet sonuçlarını ekrana basıyoruz.
# Bu özet: regresyon katsayıları, p-değerleri ve R^2 gibi istatistiksel metrikler içerir.
print(results.summary())
