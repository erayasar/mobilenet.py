import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import os

# Örnek veri oluşturma
# X verisi: Görüntülerin piksel değerleri
# y verisi: Etiketler (0: normal, 1: çıplak)
nudity_dir ="./data/nudity"
nomral_dir ="./data/normal"
nudity_files = os.listdir(nudity_dir)
normal_files =os.listdir(nomral_dir)
num_samples = 100
image_shape = (64, 64, 3)  # 64x64 boyutunda renkli görüntüler
X = np.random.rand(num_samples, *image_shape)  # 100 adet görüntü oluştur
y = np.random.randint(2, size=(num_samples,))  # 100 adet etiket (0 veya 1)

# Verileri eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid aktivasyonu ile çıktı layer'ı
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Çünkü iki sınıf var (çıplak, normal)
              metrics=['accuracy'])

# Modeli eğitme
epochs = 20
batch_size = 16
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Modelin performansını değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Modeli görselleştirme
plot_model(model, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)
