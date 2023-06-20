import tensorflow as tf
import numpy as np

# 数据加载，按照8:2的比例加载花卉数据
from sklearn.metrics import classification_report


def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


def test(is_transfer=True):
    train_ds, val_ds, class_names = data_load("./flower_photos/flower_photos", 224, 224, 4)
    if is_transfer == 0:
        model = tf.keras.models.load_model("models/mobilenet_flower.h5")
    elif is_transfer == 1:
        model = tf.keras.models.load_model("models/cnn_flower.h5")
    elif is_transfer == 2:
        model = tf.keras.models.load_model("models/restnet_flower.h5")
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    label =['daisy','dandelion','roses','sunflowers','tulips']
    y_pred = model.predict(val_ds)  # 对测试集进行预测
    print('y：', y_pred)
    y_pred = tf.keras.utils.to_categorical(y_pred, 10)
    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    y_pred = np.rint(y_pred)  # 四舍五入取整
    print(classification_report(label, y_pred))  # 生成预测召回率、准确率、F1值。
    print('Test accuracy :', accuracy)


if __name__ == '__main__':
    test(0)
    test(1)
    test(2)

