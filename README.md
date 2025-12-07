# 🔍 Anomalix - Detekcja Obiektów i Anomalii w Obrazach RTG

System wykrywania zagrożeń w skanach bagażu/pojazdów. Dwa podejścia: **YOLO** (konkretne obiekty) i **Autoencoder** (ogólne anomalie).
Przed przejrzeniem kodu zachęcamy do zobaczenia prezentacji, która wprowadzi temat.

## 🚀 Szybki start

### Instalacja
```bash
pip install ultralytics opencv-python numpy matplotlib pillow torch torchvision scikit-learn tqdm
```

### Użycie - YOLO (wykrywanie konkretnych obiektów)
```bash
# Umieść obrazy w predict/data/
python predict/predict.py
# Wyniki w predict/output/
```

---

## 📊 Co wykrywa?

### YOLO (YOLOv8n) - 6 klas obiektów
Butelki, Pudełka, Skrzynie, Granaty, Nożyce, Łomy

### Autoencoder
Wszystko, co odbiega od "normalnych" obrazów (bez oznaczania klas)

## ⚖️ Kiedy czego użyć?

| Użyj YOLO gdy... | Użyj Autoencoder gdy... |
|------------------|-------------------------|
| ✅ Wiesz czego szukasz | ✅ Szukasz "czegoś nietypowego" |
| ✅ Masz oznaczone dane | ✅ Brak oznaczonych anomalii |
| ✅ Potrzebujesz nazw obiektów | ✅ Potrzebujesz mapy cieplnej |
| ✅ Czas: ~20ms/obraz | ⏱️ Czas: ~100ms/obraz |

## 📁 Pliki projektu

```
yolo.ipynb                  # Pipeline YOLO (supervised)
conv_autoencoder.ipynb      # Pipeline Autoencoder (unsupervised)
predict/predict.py          # Skrypt batch prediction
model/weights/best.pt       # Wytrenowany model YOLO
```


## 👥 Autorzy

Projekt Anomalix - Detekcja obiektów i anomalii w obrazach rentgenowskich autorstwa C(offe)++

---

