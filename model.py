import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape=(224, 224, 3), num_classes=67, dropout_rate=0.5):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(dropout_rate))               # Overfitting azaltıcı
    model.add(Dense(256, activation='relu'))       # Fully Connected
    model.add(Dense(num_classes, activation='softmax'))  # Çıkış katmanı

    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()