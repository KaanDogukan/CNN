import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )

    val_gen = test_val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )

    test_gen = test_val_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )

    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    train_dir = "dataset/train"
    val_dir   = "dataset/val"
    test_dir  = "dataset/test"

    train_gen, val_gen, test_gen = get_data_generators(train_dir, val_dir, test_dir)

    print("Veri yükleyici başarıyla çalıştı.")
    print(f"Train örnek sayısı: {train_gen.samples}")
    print(f"Val örnek sayısı: {val_gen.samples}")
    print(f"Test örnek sayısı: {test_gen.samples}")
