#!/usr/bin/env python3
"""Test modelu YOLO - predykcja wszystkich plików z folderu."""

from ultralytics import YOLO
import cv2
import sys
import os
from pathlib import Path
import numpy as np

# Import funkcji przycinania z cut.py
from cut import crop_white_background


def predict_image(model_path, image_path, output_path='result.jpg', conf_threshold=0.25):
    """Wykryj obiekty na pojedynczym zdjęciu."""
    
    # Załaduj model
    if not os.path.exists(model_path):
        print(f"Błąd: Brak modelu w {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Błąd: Brak zdjęcia {image_path}")
        return
    
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Zdjęcie: {image_path}")
    print(f"Próg pewności: {conf_threshold}")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # Wczytaj obraz
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return 0
    
    # Przytnij białe tło
    img_cropped = crop_white_background(img_original)
    
    # Konwertuj z powrotem na BGR dla YOLO i rysowania
    if len(img_cropped.shape) == 2:
        img = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)
    else:
        img = img_cropped
    
    # Detekcja
    # Detekcja na przyciętym obrazie
    results = model.predict(
        source=img,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    result = results[0]
    boxes = result.boxes
    
    print(f"\nWykryto {len(boxes)} obiektów:\n")
    
    # Rysuj każdy bbox
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        class_name = model.names[cls]
        print(f"  {i+1}. {class_name}: {conf:.2%} pewności")
        print(f"     Pozycja: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
        
        # Rysuj prostokąt
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 255, 0), 3)
        
        # Etykieta
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Tło dla tekstu
        cv2.rectangle(img, (int(x1), int(y1)-label_size[1]-10), 
                     (int(x1)+label_size[0], int(y1)), (0, 255, 0), -1)
        
        # Tekst
        cv2.putText(img, label, (int(x1), int(y1)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    if len(boxes) == 0:
        print("  Brak detekcji")
    
    # Zapisz wynik
    cv2.imwrite(output_path, img)
    print(f"✓ Wynik zapisany: {output_path}")
    print("=" * 60)
    
    return len(boxes)


def main():
    # Konfiguracja
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'data'
    output_dir = script_dir / 'output'
    model_path = script_dir.parent / 'model' / 'weights' / 'best.pt'
    
    # Próg pewności (można zmienić)
    conf = 0.4
    
    # Utwórz katalog wyjściowy
    output_dir.mkdir(exist_ok=True)
    
    # Znajdź wszystkie obrazy w katalogu data
    image_extensions = ['*.bmp', '*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
    
    if not image_files:
        print(f"Brak obrazów w katalogu {input_dir}")
        return
    
    print("=" * 60)
    print(f"Znaleziono {len(image_files)} obrazów do przetworzenia")
    print(f"Model: {model_path}")
    print(f"Próg pewności: {conf}")
    print("=" * 60)
    
    # Przetwórz wszystkie obrazy
    total_detections = 0
    processed = 0
    
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Przetwarzanie: {image_path.name}")
        
        basename = image_path.stem
        output_path = output_dir / f"{basename}_result.jpg"
        
        try:
            detections = predict_image(str(model_path), str(image_path), str(output_path), conf)
            total_detections += detections
            processed += 1
        except Exception as e:
            print(f"Błąd podczas przetwarzania {image_path.name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"PODSUMOWANIE:")
    print(f"  Przetworzonych obrazów: {processed}/{len(image_files)}")
    print(f"  Łączna liczba detekcji: {total_detections}")
    print(f"  Wyniki zapisane w: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
