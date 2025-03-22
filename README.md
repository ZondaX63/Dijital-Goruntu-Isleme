Bu doküman, temel görüntü işleme kavramlarını, tekniklerini ve Python ile OpenCV kullanarak gerçekleştirilen uygulamalarını detaylı bir şekilde açıklamaktadır. Görüntü işleme, dijital görüntülerin analizi, manipülasyonu ve dönüştürülmesi ile ilgilenen bir bilgisayar bilimi alanıdır.

## İçindekiler

1. [Görüntü Yükleme ve Gösterme](#1-görüntü-yükleme-ve-gösterme)
2. [Renk Uzayları ve Dönüşümler](#2-renk-uzayları-ve-dönüşümler)
3. [Görüntü Kırpma ve Piksel Manipülasyonu](#3-görüntü-kırpma-ve-piksel-manipülasyonu)
4. [Histogram Analizi](#4-histogram-analizi)
5. [Histogram Eşitleme](#5-histogram-eşitleme)
6. [Temel Görüntü İşleme Teknikleri](#6-temel-görüntü-işleme-teknikleri)
7. [Filtreleme ve Kenar Algılama](#7-filtreleme-ve-kenar-algılama)
8. [Eşikleme Yöntemleri](#8-eşikleme-yöntemleri)
9. [Geometrik Dönüşümler](#9-geometrik-dönüşümler)
10. [Görüntü Üzerine Çizim İşlemleri](#10-görüntü-üzerine-çizim-işlemleri)
11. [Morfolojik İşlemler](#11-morfolojik-işlemler)
12. [Görüntü Segmentasyonu](#12-görüntü-segmentasyonu)

## 1. Görüntü Yükleme ve Gösterme

Görüntü işleme çalışmalarına başlamak için, öncelikle bir görüntüyü yüklemek ve görselleştirmek gerekir. OpenCV ve Matplotlib kütüphaneleri, bu işlemler için yaygın olarak kullanılır.

### Teorik Bilgi

- **OpenCV** (Open Source Computer Vision Library): Görüntü işleme ve bilgisayarlı görü uygulamaları için optimize edilmiş açık kaynaklı bir kütüphanedir.
- **Matplotlib**: Veri görselleştirme için kullanılan bir Python kütüphanesidir ve görüntüleri farklı formatlarda göstermek için kullanılabilir.
- **BGR ve RGB Renk Formatları**: OpenCV, görüntüleri varsayılan olarak BGR (Mavi-Yeşil-Kırmızı) formatında yükler, ancak çoğu görüntü işleme kütüphanesi RGB (Kırmızı-Yeşil-Mavi) formatını kullanır.

### Kod Örneği

```python
import cv2
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("resim.jpg")  # Görüntü dosyasının adını kendi dosya adınızla değiştirin

# OpenCV varsayılan olarak BGR formatında açar, bunu RGB'ye çevirelim
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Matplotlib ile görüntüyü göster
plt.imshow(image_rgb)
plt.axis("off")  # Eksenleri kaldır
plt.title("Matplotlib ile Görüntü Gösterme")
plt.show()

# OpenCV ile görüntüyü göster
cv2.imshow("OpenCV ile Görüntü", image)
cv2.waitKey(0)  # Bir tuşa basılana kadar bekle
cv2.destroyAllWindows()  # Pencereyi kapat
```

### İpuçları

- OpenCV, varsayılan olarak görüntüleri BGR formatında yükler. Matplotlib ile görselleştirmek için RGB'ye dönüştürmeyi unutmayın.
- `cv2.waitKey(0)` komutu, bir tuşa basılana kadar bekler.
- Jupyter Notebook'ta çalışırken Matplotlib genellikle daha kullanışlıdır.

## 2. Renk Uzayları ve Dönüşümler

Renk uzayları, renkleri temsil etmek için kullanılan matematiksel modellerdir. Farklı renk uzayları, farklı analizler ve işlemler için avantaj sağlar.

### Teorik Bilgi

- **RGB (Kırmızı-Yeşil-Mavi)**: Kırmızı, yeşil ve mavi kanalların kombinasyonlarıyla renkleri tanımlar.
- **Grayscale (Gri Tonlama)**: Her piksele tek bir değer atanır (genellikle 0-255), siyahtan beyaza giden bir skala oluşturur.
- **HSV (Ton-Doygunluk-Değer)**: Renkleri, ton (0-360°), doygunluk (0-100%) ve değer/parlaklık (0-100%) olarak tanımlar.
- **LAB**: Algısal olarak düzgün bir renk uzayıdır, L* parlaklık, a* kırmızı/yeşil, b* sarı/mavi bileşenlerini ifade eder.
- **YCrCb**: Dijital video işleme için kullanılır, Y parlaklık, Cr ve Cb renk bileşenleridir.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("resim.jpg")

# Farklı renk uzaylarına dönüştürme
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Görüntüleri gösterme
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# RGB
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("RGB Renk Uzayı")
axes[0, 0].axis("off")

# Grayscale
axes[0, 1].imshow(image_gray, cmap="gray")
axes[0, 1].set_title("Grayscale (Gri Tonlama)")
axes[0, 1].axis("off")

# HSV
axes[0, 2].imshow(image_hsv)
axes[0, 2].set_title("HSV (Ton, Doygunluk, Değer)")
axes[0, 2].axis("off")

# LAB
axes[1, 0].imshow(image_lab)
axes[1, 0].set_title("LAB Renk Uzayı")
axes[1, 0].axis("off")

# YCrCb
axes[1, 1].imshow(image_ycrcb)
axes[1, 1].set_title("YCrCb Renk Uzayı")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
```

### Uygulama Alanları

- **RGB**: İnsan gözüne uygun olması sebebiyle genel görüntü işlemede kullanılır.
- **Grayscale**: Kenar algılama, şekil analizi ve diğer temel görüntü işleme tekniklerinde kullanılır.
- **HSV**: Renk tabanlı nesne tanıma ve segmentasyonda etkilidir.
- **LAB**: Renk manipülasyonu ve resim işlemede kullanılır, özellikle renk düzeltmelerinde faydalıdır.
- **YCrCb**: Video sıkıştırma ve cilt rengi tespiti gibi alanlarda kullanılır.

## 3. Görüntü Kırpma ve Piksel Manipülasyonu

Görüntüler, aslında piksel değerlerinin saklandığı matrislerdir. Bu matrislerin belirli bölgelerini çıkararak (kırparak) ya da piksel değerlerini değiştirerek görüntüler üzerinde çeşitli işlemler yapabiliriz.

### Teorik Bilgi

- **Piksel**: Görüntünün en küçük birimidir, içerisinde renk bilgisi taşır.
- **Piksel Erişimi**: Python'da dizi indeksleme ile (y, x) formatında erişilir.
- **Kırpma (Cropping)**: Görüntünün belirli bir kısmını seçme işlemidir.
- **Piksel Manipülasyonu**: Piksel değerlerinin değiştirilmesi işlemidir.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("resim.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Kırpma için koordinatlar
y1, y2 = 50, 200  # Yükseklik aralığı
x1, x2 = 100, 300  # Genişlik aralığı

# Belirli bir bölgeyi kırp
cropped_part = image_rgb[y1:y2, x1:x2]

# Kırpılan bölgenin matris temsilini göster
print("Kırpılan Bölgenin Matris Temsili:")
print(cropped_part)

# Kırpılan bölgeyi görselleştir
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cropped_part)
plt.title("Kırpılan Bölge")
plt.axis("off")

plt.tight_layout()
plt.show()

# Küçük bir bölgeyi seç ve piksel değerlerini göster
small_region_y1, small_region_y2 = 50, 60
small_region_x1, small_region_x2 = 100, 110
small_region = image_rgb[small_region_y1:small_region_y2, small_region_x1:small_region_x2]

print("\nKüçük Bölgenin Piksel Değerleri:")
print(small_region)

# Gri tonlamada aynı bölgeyi göster
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small_region_gray = gray_image[small_region_y1:small_region_y2, small_region_x1:small_region_x2]

print("\nGri Tonlamalı Bölgenin Piksel Değerleri:")
print(small_region_gray)
```

### Görüntü Kırpma Uygulamaları

- Nesne tespiti sonrası tespit edilen nesnenin çıkarılması
- İlgili alanların analizi (örneğin, bir tıbbi görüntüde belirli bir bölge)
- Veri ön işleme (örneğin, makine öğrenmesi için görüntü hazırlama)

## 4. Histogram Analizi

Histogram, bir görüntüdeki piksel değerlerinin dağılımını gösteren grafiktir. Görüntünün kontrast, parlaklık ve genel ton dağılımı hakkında bilgi verir.

### Teorik Bilgi

- **Histogram**: Piksel yoğunluk değerlerinin (0-255) frekans dağılımını gösteren grafiktir.
- **Gri Ton Histogramı**: Tek kanallı görüntülerde yoğunluk dağılımını gösterir.
- **Renkli Görüntü Histogramı**: Her renk kanalı (genellikle R, G, B) için ayrı histogramlar oluşturulur.
- **Histogram Analizi**: Görüntünün kontrastı, parlaklığı ve kalitesi hakkında bilgi sağlar.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle ve gri tonlamaya çevir
image = cv2.imread("resim.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gri tonlama histogramını hesapla
hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Histogramı çiz
plt.figure(figsize=(10, 5))
plt.plot(hist_gray, color='black')
plt.title("Gri Tonlama Histogramı")
plt.xlabel("Piksel Değeri (0-255)")
plt.ylabel("Frekans")
plt.xlim([0, 256])
plt.grid()
plt.show()

# Renkli görüntü için kanal bazlı histogramlar
colors = ('b', 'g', 'r')  # OpenCV'de BGR sıralaması var
channel_labels = ['Mavi', 'Yeşil', 'Kırmızı']

plt.figure(figsize=(10, 5))
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=channel_labels[i])

plt.title("Renkli Görüntü Histogramı")
plt.xlabel("Piksel Değeri (0-255)")
plt.ylabel("Frekans")
plt.legend()
plt.grid()
plt.show()
```

### Histogram Analizi İpuçları

- Görüntü çok karanlıksa, histogram sol tarafa kayar.
- Görüntü çok parlaksa, histogram sağ tarafa kayar.
- İyi kontrastlı bir görüntüde histogram genellikle tüm değer aralığına dağılır.
- Histogram eşitleme, düşük kontrastlı görüntüleri iyileştirmek için kullanılır.

## 5. Histogram Eşitleme

Histogram eşitleme, görüntünün kontrastını artırmak için kullanılan bir tekniktir. Piksel değerlerinin dağılımını daha dengeli hale getirerek daha iyi bir görsel kalite sağlar.

### Teorik Bilgi

- **Histogram Eşitleme**: Görüntü piksellerinin dağılımını daha geniş bir aralığa yayarak kontrastı artıran bir tekniktir.
- **Klasik Histogram Eşitleme**: Tüm görüntüye global olarak uygulanan eşitleme yöntemidir.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Görüntüyü küçük bölgelere ayırarak lokal eşitleme yapan ve sınırlandırılmış kontrast sağlayan gelişmiş bir yöntemdir.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle ve gri tonlamaya çevir
image = cv2.imread("resim.jpg", cv2.IMREAD_GRAYSCALE)

# Klasik Histogram Eşitleme
equalized = cv2.equalizeHist(image)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_equalized = clahe.apply(image)

# Histogramları hesapla
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
hist_clahe = cv2.calcHist([clahe_equalized], [0], None, [256], [0, 256])

# Sonuçları görselleştir
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# Orijinal Görüntü
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("Orijinal Görüntü")
axes[0, 0].axis("off")

# Orijinal Histogram
axes[0, 1].plot(hist_original, color='black')
axes[0, 1].set_title("Orijinal Histogram")

# Klasik Histogram Eşitleme Sonucu
axes[1, 0].imshow(equalized, cmap="gray")
axes[1, 0].set_title("Klasik Histogram Eşitleme")
axes[1, 0].axis("off")

# Klasik Histogram Eşitleme Histogramı
axes[1, 1].plot(hist_equalized, color='black')
axes[1, 1].set_title("Klasik Histogram Eşitleme Histogramı")

# CLAHE Histogram Eşitleme Sonucu
axes[2, 0].imshow(clahe_equalized, cmap="gray")
axes[2, 0].set_title("CLAHE Histogram Eşitleme")
axes[2, 0].axis("off")

# CLAHE Histogramı
axes[2, 1].plot(hist_clahe, color='black')
axes[2, 1].set_title("CLAHE Histogramı")

plt.tight_layout()
plt.show()
```

### Histogram Eşitleme Karşılaştırması

- **Klasik Histogram Eşitleme**: Tüm görüntüdeki piksel değerlerini global olarak eşitler.

  - Avantaj: Basit ve hızlı bir yöntem.
  - Dezavantaj: Görüntünün bazı bölgelerinde aşırı kontrast veya gürültü artışı olabilir.
- **CLAHE**: Görüntüyü küçük kareler şeklinde bölgelere ayırarak, her bölge için ayrı histogram eşitleme uygular.

  - Avantaj: Lokal kontrastı iyileştirir ve gürültüyü sınırlar.
  - Dezavantaj: Klasik yönteme göre daha yavaştır.

## 6. Temel Görüntü İşleme Teknikleri

Temel görüntü işleme teknikleri, görüntülerin manipülasyonu ve analizinde sıkça kullanılan basit ama etkili yöntemlerdir.

### Teorik Bilgi

- **Gri Tonlama**: Renkli bir görüntüyü, her pikselin tek bir yoğunluk değerine sahip olduğu gri ton görüntüsüne dönüştürme.
- **Yeniden Boyutlandırma**: Görüntünün boyutlarını değiştirme (küçültme veya büyütme).
- **Döndürme**: Görüntüyü belirli bir açıyla döndürme.
- **Aynalama**: Görüntüyü yatay, dikey veya her iki eksende çevirme.
- **Bulanıklaştırma**: Görüntüdeki gürültüyü azaltmak veya detayları yumuşatmak için kullanılır.
- **Kenar Algılama**: Görüntüdeki nesnelerin kenarlarını belirlemek için kullanılır.
- **Histogram Eşitleme**: Görüntünün kontrastını artırmak için kullanılır.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("resim.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Gri Tonlama (Grayscale)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Yeniden Boyutlandırma (Resize)
new_width, new_height = 300, 300
image_resized = cv2.resize(image_rgb, (new_width, new_height))

# 3. Döndürme (Rotate)
(h, w) = image.shape[:2]  # Yükseklik ve genişlik
center = (w // 2, h // 2)  # Döndürme merkezi
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 derece döndürme
image_rotated = cv2.warpAffine(image_rgb, matrix, (w, h))

# 4. Aynalama (Flip)
image_flipped = cv2.flip(image_rgb, 1)  # 1: Yatay çevirme, 0: Dikey çevirme, -1: Her ikisi

# 5. Bulanıklaştırma (Blur)
image_blur = cv2.GaussianBlur(image_rgb, (15, 15), 0)

# 6. Kenar Algılama (Canny)
image_edges = cv2.Canny(image_gray, 100, 200)

# 7. Histogram Eşitleme (Contrast Enhancement)
image_equalized = cv2.equalizeHist(image_gray)

# Tüm işlemleri görselleştirme
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Orijinal Görüntü
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Orijinal Görüntü")
axes[0, 0].axis("off")

# Grayscale
axes[0, 1].imshow(image_gray, cmap="gray")
axes[0, 1].set_title("Gri Tonlama (Grayscale)")
axes[0, 1].axis("off")

# Yeniden Boyutlandırma
axes[0, 2].imshow(image_resized)
axes[0, 2].set_title("Yeniden Boyutlandırma (Resize)")
axes[0, 2].axis("off")

# Döndürme
axes[0, 3].imshow(image_rotated)
axes[0, 3].set_title("Döndürme (Rotate 45°)")
axes[0, 3].axis("off")

# Aynalama
axes[1, 0].imshow(image_flipped)
axes[1, 0].set_title("Aynalama (Flip)")
axes[1, 0].axis("off")

# Bulanıklaştırma
axes[1, 1].imshow(image_blur)
axes[1, 1].set_title("Bulanıklaştırma (Blur)")
axes[1, 1].axis("off")

# Kenar Algılama
axes[1, 2].imshow(image_edges, cmap="gray")
axes[1, 2].set_title("Kenar Algılama (Canny)")
axes[1, 2].axis("off")

# Histogram Eşitleme
axes[1, 3].imshow(image_equalized, cmap="gray")
axes[1, 3].set_title("Histogram Eşitleme")
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()
```

### Uygulama Alanları

- **Gri Tonlama**: Birçok görüntü işleme algoritması gri tonlamalı görüntüler üzerinde çalışır.
- **Yeniden Boyutlandırma**: Görüntüleri sabit boyuta getirmek, makine öğrenmesi uygulamalarında yaygındır.
- **Döndürme ve Aynalama**: Veri çoğaltma (data augmentation) tekniklerinde kullanılır.
- **Bulanıklaştırma**: Gürültü azaltma ve ön işleme için kullanılır.
- **Kenar Algılama**: Nesne tespiti, şekil analizi ve segmentasyon için temel bir adımdır.
- **Histogram Eşitleme**: Düşük kontrastlı görüntüleri iyileştirmek için kullanılır.

## 7. Filtreleme ve Kenar Algılama

Filtreleme ve kenar algılama, görüntü işlemede temel işlemlerdir. Filtreleme, görüntü üzerindeki gürültüyü azaltmak veya belirli özellikleri vurgulamak için kullanılırken, kenar algılama, görüntüdeki nesnelerin sınırlarını tespit etmek için kullanılır.

### Teorik Bilgi

#### Filtreleme Teknikleri

- **Gaussian Blur**: Görüntüyü yumuşatır, gürültüyü azaltır. Merkeze yakın piksellere daha fazla ağırlık verilir.
- **Median Blur**: Tuz ve biber gürültüsünü gidermede etkilidir. Her pikseli, etrafındaki piksellerin medyanı ile değiştirir.
- **Bilateral Filter**: Kenarları korurken gürültüyü azaltan gelişmiş bir filtredir.

#### Kenar Algılama Teknikleri

- **Sobel Operatörü**: Görüntüdeki yatay ve dikey kenarları tespit eder. Birinci türev tabanlıdır.
- **Laplacian Operatörü**: Görüntüdeki tüm yönlerdeki kenarları tespit eder. İkinci türev tabanlıdır.
- **Canny Kenar Algılama**: Çoklu aşamalı bir algoritma kullanarak kenarları tespit eder. Gürültüye karşı dayanıklıdır.
- **Prewitt Operatörü**: Sobel'e alternatif olarak kullanılan bir kenar algılama operatörüdür.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("resim.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Filtreleme Teknikleri
# 1. Gaussian Blur
image_gaussian = cv2.GaussianBlur(image_rgb, (15, 15), 0)

# 2. Median Blur
image_median = cv2.medianBlur(image_rgb, 5)

# 3. Bilateral Filter
image_bilateral = cv2.bilateralFilter(image_rgb, 9, 75, 75)

# Kenar Algılama Teknikleri
# 4. Sobel Kenar Algılama
sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

# 5. Laplacian Kenar Algılama
laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)

# 6. Canny Kenar Algılama
canny_edges = cv2.Canny(image_gray, 100, 200)

# 7. Prewitt Kenar Algılama
prewitt_x = cv2.filter2D(image_gray, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
prewitt_y = cv2.filter2D(image_gray, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
prewitt_combined = cv2.bitwise_or(prewitt_x, prewitt_y)

# Sonuçları görselleştirme
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# Orijinal Görüntü
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Orijinal Görüntü")
axes[0, 0].axis("off")

# Gaussian Blur
axes[0, 1].imshow(image_gaussian)
axes[0, 1].set_title("Gaussian Blur")
axes[0, 1].axis("off")

# Median Blur
axes[0, 2].imshow(image_median)
axes[0, 2].set_title("Median Blur")
axes[0, 2].axis("off")

# Bilateral Filter
axes[1, 0].imshow(image_bilateral)
axes[1, 0].set_title("Bilateral Filter")
axes[1, 0].axis("off")

# Sobel Kenar Algılama
axes[1, 1].imshow(sobel_combined, cmap="gray")
axes[1, 1].set_title("Sobel Kenar Algılama")
axes[1, 1].axis("off")

# Laplacian Kenar Algılama
axes[1, 2].imshow(laplacian, cmap="gray")
axes[1, 2].set_title("Laplacian Kenar Algılama")
axes[1, 2].axis("off")

# Canny Kenar Algılama
axes[2, 0].imshow(canny_edges, cmap="gray")
axes[2, 0].set_title("Canny Kenar Algılama")
axes[2, 0].axis("off")

# Prewitt Kenar Algılama
axes[2, 1].imshow(prewitt_combined, cmap="gray")
axes[2, 1].set_title("Prewitt Kenar Algılama")
axes[2, 1].axis("off")

plt.tight_layout()
plt.show()
```

### Filtreleme ve Kenar Algılama Karşılaştırması

#### Filtreleme Teknikleri

- **Gaussian Blur**: Genel gürültü azaltma için idealdir. Kenarlarda bulanıklık oluşturabilir.
- **Median Blur**: Tuz ve biber gürültüsüne karşı etkilidir. Detayları korumada orta düzeydedir.
- **Bilateral Filter**: Kenarları korurken gürültüyü azaltır. Hesaplama açısından daha yavaştır.

#### Kenar Algılama Teknikleri

- **Sobel**: Hızlı ve basittir, yönlü kenarları iyi tespit eder. Gürültüye karşı orta düzeyde hassastır.
- **Laplacian**: Tüm yönlerdeki kenarları tespit eder. Gürültüye karşı oldukça hassastır.
- **Canny**: Çoklu aşamalı bir yaklaşım kullanır, gürültüye karşı dayanıklıdır ve genellikle en iyi sonuçları verir.
- **Prewitt**: Sobel'e alternatif basit bir operatördür, daha az ağırlıklı merkez hesaplama kullanır.

## 8. Eşikleme Yöntemleri

Eşikleme (Thresholding), bir görüntüyü genellikle siyah ve beyaz (ikili) görüntüye dönüştüren bir tekniktir. Belirli bir eşik değerinin altındaki veya üstündeki pikselleri 0 veya 255 değerine atayarak görüntüyü segmente eder.

### Teorik Bilgi

- **Global Thresholding (Sabit Eşikleme)**: Tüm görüntü için tek bir eşik değeri kullanılır.
- **Otsu Thresholding**: Eşik değerini otomatik olarak hesaplar, bimodal histogramlar için idealdir.
- **Adaptive Thresholding**: Görüntüyü bölgelere ayırarak her bölge için farklı eşik değerleri hesaplar.
- **Binary Thresholding**: Piksel değeri eşik değerinden büyükse maksimum değer, değilse 0 atanır.
- **Binary Inverse Thresholding**: Binary Thresholding'in tersidir.
- **Truncate Thresholding**: Piksel değeri eşik değerinden büyükse eşik değeri, değilse orijinal değer korunur.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle ve gri tonlamaya çevir
image = cv2.imread("resim.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram analizi
hist_gray = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
colors = ('b', 'g', 'r')
hist_channels = [cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]

# 1. Global Thresholding (Sabit Eşikleme)
_, thresh_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

# 2. Otsu Thresholding
_, thresh_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 4. Binary Inverse Thresholding
_, thresh_binary_inv = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)

# 5. Truncate Thresholding
_, thresh_trunc = cv2.threshold(image_gray, 127, 255, cv2.THRESH_TRUNC)

# 6. Gaussian Adaptive Thresholding
gaussian_adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

# Sonuçları görselleştirme
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# Orijinal Görüntü
axes[0, 0].imshow(image_gray, cmap="gray")
axes[0, 0].set_title("Orijinal Gri Tonlama Görüntü")
axes[0, 0].axis("off")

# Grayscale Histogram
axes[0, 1].plot(hist_gray, color='black')
axes[0, 1].set_title("Grayscale Histogram")

# RGB Histogram
axes[0, 2].set_title("RGB Histogram")
for i, color in enumerate(colors):
    axes[0, 2].plot(hist_channels[i], color=color)
axes[0, 2].set_xlim([0, 256])

# Global Binary Thresholding
axes[1, 0].imshow(thresh_binary, cmap="gray")
axes[1, 0].set_title("Global Binary Thresholding")
axes[1, 0].axis("off")

# Otsu Thresholding
axes[1, 1].imshow(thresh_otsu, cmap="gray")
axes[1, 1].set_title("Otsu Thresholding")
axes[1, 1].axis("off")

# Adaptive Thresholding
axes[1, 2].imshow(adaptive_thresh, cmap="gray")
axes[1, 2].set_title("Adaptive Thresholding")
axes[1, 2].axis("off")

# Binary Inverse Thresholding
axes[2, 0].imshow(thresh_binary_inv, cmap="gray")
axes[2, 0].set_title("Binary Inverse Thresholding")
axes[2, 0].axis("off")

# Truncate Thresholding
axes[2, 1].imshow(thresh_trunc, cmap="gray")
axes[2, 1].set_title("Truncate Thresholding")
axes[2, 1].axis("off")

# Gaussian Adaptive Thresholding
axes[2, 2].imshow(gaussian_adaptive, cmap="gray")
axes[2, 2].set_title("Gaussian Adaptive Thresholding")
axes[2, 2].axis("off")

plt.tight_layout()
plt.show()
```

### Eşikleme Yöntemlerinin Karşılaştırması

- **Global Thresholding**: Basit ve hızlı, ancak değişken aydınlatma koşullarında iyi çalışmaz.
- **Otsu Thresholding**: Bimodal histogramlı görüntülerde en iyi sonuçları verir. Eşik değerini otomatik hesaplar.
- **Adaptive Thresholding**: Değişken aydınlatma koşullarında daha iyi sonuçlar verir. Yerel bölgelere dayalı eşikleme yapar.
- **Binary ve Binary Inverse**: Piksel değerlerini iki gruba (0 ve 255) ayırır, ikili görüntüler oluşturur.
- **Truncate**: Eşik değerinin üzerindeki pikselleri sınırlar, altındakileri korur.

## 9. Geometrik Dönüşümler

Geometrik dönüşümler, görüntülerin boyut, konum, yönelim veya şeklini değiştirmek için kullanılan tekniklerdir. Bu dönüşümler, görüntü işlemede yaygın olarak kullanılır ve çeşitli uygulamalarda önemli rol oynar.

### Teorik Bilgi

- **Boyutlandırma (Resize)**: Görüntünün boyutlarını değiştirme işlemidir.

  - **Büyütme**: Görüntünün boyutunu artırır.
  - **Küçültme**: Görüntünün boyutunu azaltır.
- **Döndürme (Rotation)**: Görüntüyü belirli bir merkez etrafında döndürme işlemidir.
- **Ölçekleme (Scaling)**: Görüntünün boyutlarını belli bir oranda değiştirme işlemidir.
- **Affine Dönüşüm**: Paralel çizgilerin paralel kaldığı, ancak açıların korunmadığı bir dönüşüm türüdür. Döndürme, ölçekleme, kaydırma ve çarpıtma işlemlerini içerebilir.
- **Perspektif Dönüşüm**: Görüntünün perspektifini değiştiren bir dönüşüm türüdür. Paralel çizgilerin paralel kalması garanti edilmez.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("resim.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Boyutlandırma (Resize) - Büyütme ve Küçültme
image_resized_up = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)  # %150 büyütme
image_resized_down = cv2.resize(image_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # %50 küçültme

# 2. Döndürme (Rotate)
(h, w) = image.shape[:2]  # Yükseklik ve genişlik
center = (w // 2, h // 2)  # Döndürme merkezi
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 derece döndürme
image_rotated = cv2.warpAffine(image_rgb, matrix, (w, h))

# 3. Affine Transform (Çarpıtma)
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # Orijinal noktalar
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])  # Hedef noktalar
matrix_affine = cv2.getAffineTransform(pts1, pts2)
image_affine = cv2.warpAffine(image_rgb, matrix_affine, (w, h))

# 4. Perspektif Dönüşümü (Perspective Transform)
pts1_persp = np.float32([[100, 100], [400, 100], [100, 400], [400, 400]])
pts2_persp = np.float32([[50, 50], [450, 50], [100, 500], [400, 450]])
matrix_perspective = cv2.getPerspectiveTransform(pts1_persp, pts2_persp)
image_perspective = cv2.warpPerspective(image_rgb, matrix_perspective, (w, h))

# Tüm dönüşümleri görselleştirme
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Orijinal Görüntü
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Orijinal Görüntü")
axes[0, 0].axis("off")

# Büyütme
axes[0, 1].imshow(image_resized_up)
axes[0, 1].set_title("Büyütülmüş Görüntü (%150)")
axes[0, 1].axis("off")

# Küçültme
axes[0, 2].imshow(image_resized_down)
axes[0, 2].set_title("Küçültülmüş Görüntü (%50)")
axes[0, 2].axis("off")

# Döndürme
axes[1, 0].imshow(image_rotated)
axes[1, 0].set_title("45 Derece Döndürme")
axes[1, 0].axis("off")

# Affine Çarpıtma
axes[1, 1].imshow(image_affine)
axes[1, 1].set_title("Affine Transform (Çarpıtma)")
axes[1, 1].axis("off")

# Perspektif Dönüşüm
axes[1, 2].imshow(image_perspective)
axes[1, 2].set_title("Perspektif Dönüşüm")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()
```

### Geometrik Dönüşümlerin Uygulama Alanları

- **Boyutlandırma**: Görüntüleri farklı ekran boyutlarına uyarlamak veya makine öğrenmesi modelleri için standart boyuta getirmek için kullanılır.
- **Döndürme**: Görüntüleri doğru yönelimde göstermek veya veri çoğaltma için kullanılır.
- **Affine Dönüşüm**: Görüntü hizalama, görüntü kayıt (registration) ve basit 3D efektleri için kullanılır.
- **Perspektif Dönüşüm**: Görüntünün bakış açısını değiştirme, belge tarama ve 3D rekonstrüksiyon gibi uygulamalarda kullanılır.

## 10. Görüntü Üzerine Çizim İşlemleri

OpenCV, görüntüler üzerine çeşitli şekiller, çizgiler ve metin eklemek için bir dizi fonksiyon sunar. Bu fonksiyonlar, görüntü analizi sonuçlarını görselleştirmek veya görüntü üzerine ek bilgiler eklemek için kullanılır.

### Teorik Bilgi

- **Çizgi Çizme**: İki nokta arasına doğru çizgi çizer.
- **Dikdörtgen Çizme**: İki köşe noktası belirtilerek dikdörtgen oluşturur.
- **Daire Çizme**: Merkez ve yarıçap belirtilerek daire çizer.
- **Elips Çizme**: Merkez, eksen uzunlukları ve dönme açısı belirtilerek elips çizer.
- **Çokgen (Poligon) Çizme**: Noktaların dizisi ile çokgen oluşturur.
- **Metin Ekleme**: Belirtilen konuma yazı ekler.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("resim.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Çizgi Çizme
image_line = image_rgb.copy()
cv2.line(image_line, (50, 50), (300, 300), (255, 0, 0), 5)  # Mavi, 5 kalınlık

# 2. Dikdörtgen Çizme
image_rectangle = image_rgb.copy()
cv2.rectangle(image_rectangle, (50, 50), (300, 300), (0, 255, 0), 3)  # Yeşil, 3 kalınlık

# 3. Daire Çizme
image_circle = image_rgb.copy()
cv2.circle(image_circle, (200, 200), 100, (0, 0, 255), -1)  # Kırmızı, içi dolu

# 4. Elips Çizme
image_ellipse = image_rgb.copy()
cv2.ellipse(image_ellipse, (250, 250), (100, 50), 45, 0, 360, (255, 255, 0), 2)  # Sarı, 2 kalınlık

# 5. Çokgen (Poligon) Çizme
image_polygon = image_rgb.copy()
pts = np.array([[100, 100], [300, 50], [400, 200], [200, 300]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image_polygon, [pts], isClosed=True, color=(0, 255, 255), thickness=3)  # Açık mavi

# 6. Metin Ekleme
image_text = image_rgb.copy()
cv2.putText(image_text, "OpenCV Text!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

# Tüm işlemleri görselleştirme
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Çizgi
axes[0, 0].imshow(image_line)
axes[0, 0].set_title("Çizgi Çizme")
axes[0, 0].axis("off")

# Dikdörtgen
axes[0, 1].imshow(image_rectangle)
axes[0, 1].set_title("Dikdörtgen Çizme")
axes[0, 1].axis("off")

# Daire
axes[0, 2].imshow(image_circle)
axes[0, 2].set_title("Daire Çizme")
axes[0, 2].axis("off")

# Elips
axes[1, 0].imshow(image_ellipse)
axes[1, 0].set_title("Elips Çizme")
axes[1, 0].axis("off")

# Poligon
axes[1, 1].imshow(image_polygon)
axes[1, 1].set_title("Poligon Çizme")
axes[1, 1].axis("off")

# Metin
axes[1, 2].imshow(image_text)
axes[1, 2].set_title("Metin Ekleme")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()
```

### Çizim İşlemlerinin Kullanım Alanları

- **Nesne Tespiti**: Tespit edilen nesnelerin etrafını kutu ile işaretlemek için dikdörtgen çizme.
- **Yüz Tanıma**: Tespit edilen yüzleri daire veya dikdörtgen ile işaretlemek.
- **Landmark Tespiti**: Özel noktaları (yüz, vücut, vb.) işaretlemek için çizgiler veya daireler.
- **Bilgi Görselleştirme**: Görüntü üzerine önemli bilgileri metin olarak eklemek.
- **Grafik Çizimi**: Veri görselleştirmesi için çizgiler, daireler ve diğer şekiller kullanmak.

## 11. Morfolojik İşlemler

Morfolojik işlemler, bir görüntü üzerinde şekil veya yapı tabanlı operasyonlar gerçekleştirmek için kullanılan tekniklerdir. Bu işlemler, özellikle ikili (binary) görüntüler üzerinde, görüntüdeki nesnelerin şekillerini değiştirmek veya özelliklerini analiz etmek için kullanılır.

### Teorik Bilgi

- **Kernel (Çekirdek/Yapısal Element)**: Morfolojik işlemlerde kullanılan, genellikle küçük bir kare veya daire şeklindeki matristir.
- **Erozyon (Erosion)**: Nesnelerin sınırlarını aşındırır, küçük nesneleri ortadan kaldırır, nesneleri inceltir.
- **Genişleme/Dilasyon (Dilation)**: Nesnelerin sınırlarını genişletir, boşlukları doldurur, nesneleri kalınlaştırır.
- **Açılma (Opening)**: Erozyon ardından dilasyon uygulanır. Küçük nesneleri ve gürültüyü kaldırır, nesne sınırlarını düzgünleştirir.
- **Kapanma (Closing)**: Dilasyon ardından erozyon uygulanır. Küçük boşlukları ve çizgileri doldurur.
- **Morfololojik Gradient**: Dilasyon ve erozyonun farkıdır, nesnelerin sınırlarını vurgular.
- **Top Hat**: Orijinal görüntü ile açılma sonucu arasındaki farktır, parlak detayları bulur.
- **Black Hat**: Kapanma sonucu ile orijinal görüntü arasındaki farktır, koyu detayları bulur.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle ve gri tonlamaya çevir
image = cv2.imread("resim.jpg", cv2.IMREAD_GRAYSCALE)

# Çekirdek (Kernel) tanımlama (5x5 boyutunda)
kernel = np.ones((5, 5), np.uint8)

# Morfolojik İşlemleri Uygula
image_erosion = cv2.erode(image, kernel, iterations=1)  # Erozyon
image_dilation = cv2.dilate(image, kernel, iterations=1)  # Genişleme
image_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Açılma
image_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Kapanma
image_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)  # Gradient
image_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)  # Top Hat
image_blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)  # Black Hat

# Sonuçları görselleştirme
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Orijinal Görüntü
axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("Orijinal Görüntü")
axes[0, 0].axis("off")

# Erozyon
axes[0, 1].imshow(image_erosion, cmap="gray")
axes[0, 1].set_title("Erozyon (Erosion)")
axes[0, 1].axis("off")

# Genişleme
axes[0, 2].imshow(image_dilation, cmap="gray")
axes[0, 2].set_title("Genişleme (Dilation)")
axes[0, 2].axis("off")

# Açılma
axes[0, 3].imshow(image_opening, cmap="gray")
axes[0, 3].set_title("Açılma (Opening)")
axes[0, 3].axis("off")

# Kapanma
axes[1, 0].imshow(image_closing, cmap="gray")
axes[1, 0].set_title("Kapanma (Closing)")
axes[1, 0].axis("off")

# Gradient
axes[1, 1].imshow(image_gradient, cmap="gray")
axes[1, 1].set_title("Morphological Gradient")
axes[1, 1].axis("off")

# Top Hat
axes[1, 2].imshow(image_tophat, cmap="gray")
axes[1, 2].set_title("Top Hat")
axes[1, 2].axis("off")

# Black Hat
axes[1, 3].imshow(image_blackhat, cmap="gray")
axes[1, 3].set_title("Black Hat")
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()
```

### Morfolojik İşlemlerin Uygulama Alanları

- **Gürültü Temizleme**: Opening (açılma) işlemi ile görüntüdeki küçük noktalar temizlenebilir.
- **Boşluk Doldurma**: Closing (kapanma) işlemi ile nesnelerdeki küçük boşluklar kapatılabilir.
- **Nesne Tespiti**: Morfolojik işlemler ile nesneler belirgin hale getirilebilir.
- **İskelet Çıkarma**: Erozyon uygulanarak nesnenin iskeleti çıkarılabilir.
- **Özellik Ayıklama**: Top Hat ve Black Hat işlemleri ile parlak veya koyu detaylar çıkarılabilir.

# 12. Görüntü Segmentasyonu

Görüntü segmentasyonu, bir görüntüyü anlamlı bölgelere ayırma işlemidir. Bu işlem, görüntüdeki nesneleri veya bölgeleri belirleme ve ayırma amacıyla kullanılır.

### Teorik Bilgi

- **Segmentasyon**: Görüntüyü anlamlı parçalara (nesneler, bölgeler vb.) ayırma işlemidir.
- **Watershed Algoritması**: Su havzası analizine dayalı bir segmentasyon algoritmasıdır. Görüntüyü, "su havzaları" ve "sırtlar" olarak modelleyerek bölgelere ayırır.
- **Mesafe Dönüşümü (Distance Transform)**: Her pikselin en yakın sınır pikselinden olan mesafesini hesaplar.
- **Marker**: Watershed algoritmasında başlangıç noktalarını belirlemek için kullanılan işaretçilerdir.

### Kod Örneği

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükleme
image = cv2.imread("rice_grain.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Gri tonlamaya çevirme
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Gürültü azaltma için bulanıklaştırma (Gaussian Blur)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Eşikleme (Thresholding)
_, thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_OTSU)

# 4. Morfolojik Açılma (Küçük gürültüleri temizleme)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

# 5. Arka plan maskesi (Genişleme ile)
sure_bg = cv2.dilate(opening, kernel, iterations=2)

# 6. Ön plan maskesi (Mesafe dönüşümü ile)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

# 7. Bilinmeyen bölge belirleme (Arka plan - Ön plan)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 8. Marker etiketleme
_, markers = cv2.connectedComponents(sure_fg)

# Arka planı 1 değil -1 yaparak Watershed algoritması için ayarlıyoruz
markers = markers + 1
markers[unknown == 255] = 0

# 9. Watershed Algoritmasını Çalıştır
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]  # Kenarları kırmızı renkte çiz

# Sonuçları görselleştirme
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Orijinal Görüntü
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Orijinal Görüntü")
axes[0, 0].axis("off")

# Gri Tonlama
axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Gri Tonlama")
axes[0, 1].axis("off")

# Threshold
axes[0, 2].imshow(thresh, cmap="gray")
axes[0, 2].set_title("Thresholding")
axes[0, 2].axis("off")

# Açılma (Morphology)
axes[0, 3].imshow(opening, cmap="gray")
axes[0, 3].set_title("Morfolojik Açılma")
axes[0, 3].axis("off")

# Arka Plan
axes[1, 0].imshow(sure_bg, cmap="gray")
axes[1, 0].set_title("Arka Plan Maskesi")
axes[1, 0].axis("off")

# Ön Plan
axes[1, 1].imshow(sure_fg, cmap="gray")
axes[1, 1].set_title("Ön Plan Maskesi")
axes[1, 1].axis("off")

# Watershed Sonucu
axes[1, 2].imshow(markers, cmap="jet")
axes[1, 2].set_title("Watershed Sonucu (Etiketler)")
axes[1, 2].axis("off")

# Segmentasyon Sonucu
axes[1, 3].imshow(image)
axes[1, 3].set_title("Watershed Sonuçlu Görüntü")
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()

# Nesne sayısını hesaplama
num_labels, markers = cv2.connectedComponents(sure_fg)
nesne_sayisi = num_labels - 1  # Arka plan hariç gerçek nesne sayısı
print(f"Tespit edilen nesne sayısı: {nesne_sayisi}")
```

### Watershed Algoritmasının Adımları

1. **Gri Tonlamaya Çevirme**: Renkli görüntüyü gri tonlamalı görüntüye dönüştürme.
2. **Gürültü Azaltma**: Gürültüyü azaltmak için görüntüye Gaussian Blur uygulama.
3. **Eşikleme (Thresholding)**: Görüntüyü ikili (binary) formata dönüştürme.
4. **Morfolojik Açılma**: Küçük gürültüleri temizlemek için erozyon ve dilasyon işlemlerini uygulama.
5. **Arka Plan Maskesi Oluşturma**: Genişleme (dilasyon) uygulayarak arka plan maskesi oluşturma.
6. **Ön Plan Maskesi Oluşturma**: Mesafe dönüşümü ve eşikleme ile ön plan bölgelerini belirleme.
7. **Bilinmeyen Bölge Belirleme**: Arka plan ve ön plan arasında kalan bölgeleri tespit etme.
8. **Marker Etiketleme**: Ön plan bölgelerini etiketleme ve bilinmeyen bölgeleri 0 olarak işaretleme.
9. **Watershed Algoritması Uygulama**: Watershed algoritmasını çalıştırarak segmentasyon sonucunu elde etme.

### Segmentasyon Uygulamaları

- **Tıbbi Görüntüleme**: Hücre, tümör veya organ segmentasyonu.
- **Nesne Sayma**: Fotoğraftaki belirli nesnelerin sayısını belirleme.
- **Nesne Tanıma**: Görüntüdeki nesneleri tanımlamak için ön işleme adımı olarak kullanılır.
- **Otonom Sürüş**: Yol, araç, yaya gibi öğeleri ayırt etmek için kullanılır.
- **Üretim Kalite Kontrolü**: Kusur tespiti ve parça sayımı.

### Watershed Algoritmasının Avantajları ve Dezavantajları

#### Avantajlar

- Kapalı konturlar oluşturur.
- Kenarları belirgin ve tutarlı bir şekilde tespit eder.
- Birbirine çok yakın nesneleri ayırt edebilir.

#### Dezavantajlar

- Aşırı segmentasyon (over-segmentation) problemi olabilir.
- Uygun markerların belirlenmesi genellikle zordur.
- Gürültülü görüntülerde performansı düşebilir.

### Gelişmiş Segmentasyon Teknikleri

1. **K-Means Segmentasyon**: Pikselleri renk veya yoğunluk değerlerine göre kümelere ayırır.
2. **Mean Shift Segmentasyon**: Benzer özelliklere sahip pikselleri kümeleme tabanlı bir yöntemdir.
3. **Grabcut Algoritması**: Kullanıcı etkileşimli segmentasyon algoritmasıdır, ön ve arka plan belirlenir.
4. **Derin Öğrenme Tabanlı Segmentasyon**:

   - U-Net
   - Mask R-CNN
   - DeepLab
   - SegNet
5. **Aktif Kontur Modelleri (Snakes)**: Nesne sınırlarını enerji minimizasyonu ile bulmaya çalışır.

### Segmentasyon Performans Değerlendirmesi

- **IoU (Intersection over Union)**: Tahmin edilen segmentasyon maskesi ile gerçek maske arasındaki kesişim/birleşim oranı.
- **Dice Katsayısı**: İki segmentasyon arasındaki benzerliği ölçer.
- **Piksel Doğruluğu**: Doğru sınıflandırılan piksellerin oranı.
- **Hassasiyet (Precision) ve Duyarlılık (Recall)**: Sınıflandırma performansını ölçen metrikler.

### Görüntü Segmentasyonu için İpuçları

1. **Ön İşleme Önemli**: Gürültü azaltma, kontrastı arttırma gibi ön işleme adımları segmentasyon performansını iyileştirir.
2. **Uygun Algoritma Seçimi**: Segmentasyon probleminin tipine ve veri setine uygun algoritma seçilmelidir.
3. **Parametre Optimizasyonu**: Segmentasyon algoritmasının parametrelerini problem özelinde optimize etmek önemlidir.
4. **Çoklu Segmentasyon Yaklaşımları**: Tek bir algoritma yerine, farklı algoritmaların sonuçlarını birleştirmek daha iyi sonuçlar verebilir.
5. **Yerel ve Global Bilginin Kullanımı**: Hem pikselin yerel özellikleri hem de görüntünün global yapısı segmentasyon için kullanılmalıdır.

## Özetle

Bu çalışma dokümanında, temel görüntü işleme konuları detaylı bir şekilde ele alınmıştır. OpenCV kullanarak görüntü yükleme ve gösterme işlemlerinden başlayarak, renk uzayları, histogramlar, filtreleme, kenar algılama, geometrik dönüşümler, morfolojik işlemler ve segmentasyon gibi temel teknikler örneklerle açıklanmıştır.

Görüntü işleme, makine öğrenmesi ve yapay zeka uygulamalarının önemli bir ön adımı olarak, gelişen teknolojilerle birlikte önemi giderek artmaktadır. Bu dokümanda sunulan temel teknikler, daha karmaşık görüntü işleme ve bilgisayarlı görü uygulamaları için sağlam bir temel oluşturmaktadır.
