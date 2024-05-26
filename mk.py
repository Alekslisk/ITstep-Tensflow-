import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Загрузка предобученной модели MobileNetV2
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Функция для предобработки изображения
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

# Функция для классификации изображения
def classify_image(image_path):
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1], decoded_predictions[0][0][2]

# Функция для выбора изображения и отображения результата
def select_image():
    path = filedialog.askopenfilename()
    if path:
        try:
            image = Image.open(path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)

            img_label.config(image=photo)
            img_label.image = photo

            class_name, confidence = classify_image(path)
            result_text.set(f"Класс: {class_name}, Доверие: {confidence:.2f}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать изображение: {e}")

# Создание основного окна
root = tk.Tk()
root.title("Image Recognition App")
root.geometry("600x600")

# Создание фрейма для размещения элементов
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Метка для отображения изображения
img_label = tk.Label(frame)
img_label.pack(pady=10)

# Кнопка для выбора изображения
btn = tk.Button(frame, text="Выбрать изображение", command=select_image)
btn.pack(pady=10)

# Текстовое поле для отображения результата классификации
result_text = tk.StringVar()
result_label = tk.Label(frame, textvariable=result_text, font=("Helvetica", 14))
result_label.pack(pady=10)

# Запуск основного цикла
root.mainloop()
