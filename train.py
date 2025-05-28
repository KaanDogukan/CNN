from model import create_model
from data_loader import get_data_generators
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Parametreler
input_shape = (224, 224, 3)
num_classes = 67
epochs = 5
learning_rates = [0.01, 0.001, 0.0001]
batch_size = 32
dropout_rates = [0.2, 0.3, 0.5, 0.7]

# Veri klasörleri
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Sonuçları kaydetmek için
results = []

# 3 learning rate ve 4 dropout için döngü
def run_experiments():
    for lr in learning_rates:
        train_gen, val_gen, test_gen = get_data_generators(
            train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=batch_size)

        for dropout_rate in dropout_rates:
            print(f"\n\n===== Training: LR={lr}, Dropout={dropout_rate} =====")

            model = create_model(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate)

            model.compile(optimizer=Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                verbose=1
            )

            loss, acc = model.evaluate(test_gen, verbose=0)
            print(f"Test Accuracy: {acc:.4f}")

            # Confusion Matrix
            Y_true = test_gen.classes
            Y_pred = model.predict(test_gen)
            Y_pred_classes = np.argmax(Y_pred, axis=1)
            cm = confusion_matrix(Y_true, Y_pred_classes)

            results.append({
                'lr': lr,
                'dropout': dropout_rate,
                'val_accuracy': history.history['val_accuracy'],
                'train_accuracy': history.history['accuracy'],
                'val_loss': history.history['val_loss'],
                'train_loss': history.history['loss'],
                'test_acc': acc,
                'confusion_matrix': cm
            })

            # Confusion matrix görseli kaydet
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
            plt.title(f"Confusion Matrix\nLR={lr}, Dropout={dropout_rate}")
            plt.savefig(f"confmat_lr{lr}_dropout{dropout_rate}.png")
            plt.close()

    return results

# Grafik çizimi fonksiyonu
def plot_validation_accuracy(results):
    plt.figure(figsize=(10,6))
    for result in results:
        label = f"LR={result['lr']}, Dropout={result['dropout']}"
        plt.plot(result['val_accuracy'], label=label)

    plt.title("Validation Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("val_accuracy_comparison.png")
    plt.show()

if __name__ == "__main__":
    results = run_experiments()
    plot_validation_accuracy(results)

    # Accuracy hesaplama örneği (elle):
    # accuracy = np.sum(Y_pred_classes == Y_true) / len(Y_true)
    # print(f"Manual Accuracy: {accuracy:.4f}")


# from model import create_model
# from data_loader import get_data_generators
# from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt

# # Parametreler
# input_shape = (224, 224, 3)
# num_classes = 67
# dropout_rate = 0.5
# batch_size = 32
# epochs = 5
# learning_rate = 0.001

# # Veriyi yükle
# train_dir = "dataset/train"
# val_dir = "dataset/val"
# test_dir = "dataset/test"
# train_gen, val_gen, test_gen = get_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=batch_size)

# # Modeli oluştur
# model = create_model(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate)

# # Derleme
# model.compile(optimizer=Adam(learning_rate=learning_rate),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Eğitim
# history = model.fit(
#     train_gen,
#     epochs=epochs,
#     validation_data=val_gen
# )

# # Grafik
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.title("Accuracy vs Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()
