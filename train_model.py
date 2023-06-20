import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# 数据加载，按照8:2的比例加载花卉数据
# 图片参数：图像高度、图像宽度、批量大小


def data_load(data_dir, img_height, img_width, batch_size):
    #划分训练集,将80%的图像用于训练
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    #划分验证集，将20%的图像用于验证
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    #在这些数据集的属性中找到类名，对应于按字母顺序排列的目录名称。
    class_names = train_ds.class_names

    return train_ds, val_ds, class_names

# def calculate_f1(y_true, y_pred):
#     y_true = tf.argmax(y_true, axis=1)
#     y_pred = tf.argmax(y_pred, axis=1)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
#     return f1

# 模型加载，指定图片处理的大小和是否进行迁移学习。3是颜色通道的RGB
def model_load(IMG_SHAPE, is_transfer):
    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    #     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # ])
    if is_transfer == 0:
        # 使用轻量型卷积神经网络
        #数据增强
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal",
                                           input_shape=(224,
                                                        224,
                                                        3)),
                # 随机旋转图像角度
                tf.keras.layers.RandomRotation(0.3),
                # 随机缩放图像大小
                tf.keras.layers.RandomZoom(0.3),
                # 随机改变图像对比度
                tf.keras.layers.RandomContrast(0.3),
            ]
        )
        # 构建MobileNet模型
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                          include_top=False,
                                          weights='imagenet')
        # base_model.trainable = True
        # fine_tune_at = 100
        for layer in base_model.layers:
            layer.trainable = False

        model = tf.keras.models.Sequential([data_augmentation,
            # 进行归一化的处理
            tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1, input_shape=IMG_SHAPE),
            base_model,
            # 对主干模型的输出进行全局平均池化
            # tf.keras.layers.GlobalAveragePooling2D(),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(5, activation='softmax')

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
    elif is_transfer == 1:
        # 数据增强。
        # 通过对已有的训练集图片 随机转换（反转、旋转、缩放等），来生成其它训练数据。
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal",
                                           input_shape=(224,
                                                        224,
                                                        3)),
                tf.keras.layers.RandomRotation(0.3),
                tf.keras.layers.RandomZoom(0.3),
                tf.keras.layers.RandomContrast(0.3),
            ]
        )
        #Sequential模型由三个卷积块 ( ) 组成，每个块中tf.keras.layers.Conv2D都有一个最大池化层 ( tf.keras.layers.MaxPooling2D)。
        # 有一个全连接层 ( tf.keras.layers.Dense)，其顶部有 128 个单元，由 ReLU 激活函数 ( 'relu') 激活。
        # 卷积层和池化层的叠加实现对输入数据的特征提取，最后连接全连接层实现分类
        # 搭建网络模型
        model = tf.keras.models.Sequential([data_augmentation,
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),#将像素的值标准化至0-1的区间内

            #  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            #  tf.keras.layers.MaxPooling2D(),
            #  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            #  tf.keras.layers.MaxPooling2D(),
            #  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            #  tf.keras.layers.MaxPooling2D(),
            #  tf.keras.layers.Dropout(0.2),
            # #从卷积层到全连接层的过渡，把多维输入一维化
            #  tf.keras.layers.Flatten(),
            #  tf.keras.layers.Dense(128, activation='relu'),
            #  tf.keras.layers.Dense(5, activation='softmax')

            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')


        ])
    #     搭建RestNet网络模型
    elif is_transfer == 2:
        # 构建数据增强模型
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal",
                                           input_shape=IMG_SHAPE),
                tf.keras.layers.RandomRotation(0.3),
                tf.keras.layers.RandomZoom(0.3),
                tf.keras.layers.RandomContrast(0.3),
            ]
        )
        # 构建ResNet模型
        base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False

        model = tf.keras.models.Sequential([
            data_augmentation,
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
            base_model,
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.GlobalAveragePooling2D(),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(5, activation='softmax')

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
    # 查看网络结构
    #模型的Model.summary方法查看网络的所有层：
    model.summary()
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    #模型训练
    # 选择Adam优化器和CategoricalCrossentropy损失函数。
    # 查看每个训练时期的训练和验证准确性，请将metrics参数传递给Model.compile.
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=METRICS)
    return model


# 展示训练过程的曲线，在训练集和验证集上创建损失和准确度图：
# def show_metrics(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     plt.figure(figsize=(8, 8))
#     plt.subplot(2, 1, 1)
#     plt.plot(acc, label='Training Accuracy')
#     plt.plot(val_acc, label='Validation Accuracy')
#     plt.legend(loc='lower right')
#     plt.ylabel('Accuracy')
#     plt.ylim([min(plt.ylim()), 1])
#     plt.title('Training and Validation Accuracy')
#
#     plt.subplot(2, 1, 2)
#     plt.plot(loss, label='Training Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel('Cross Entropy')
#     plt.ylim([min(plt.ylim()), 1])
#     plt.title('Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.show()


def show_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric
        # plt.subplot(2, 2, n + 1)
        plt.figure(n + 1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                  linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

    plt.show()



def evaluate(epochs, is_transfer):
    train_ds, val_ds, class_names = data_load("./flower_photos/flower_photos", 224, 224, 4)
    if is_transfer == 0:
        model = tf.keras.models.load_model("models/mobilenet_flower.h5")
    elif is_transfer == 1:
        model = tf.keras.models.load_model("models/cnn_flower.h5")
    elif is_transfer == 2:
        model = tf.keras.models.load_model("models/restnet_flower.h5")

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    # from sklearn.metrics import multilabel_confusion_matrix
    y_val = []
    for images, labels in val_ds:
        y_val.extend(labels.numpy())

    y_pred = model.predict(val_ds)
    # y_pred_classes = tf.argmax(y_pred, axis=1)

    y_val_oh = np.array([np.argmax(y) for y in y_val])
    y_pred_classes_oh = np.array([np.argmax(y) for y in y_pred])
    cm = confusion_matrix(y_val_oh, y_pred_classes_oh)
    print('Confusion Matrix')
    print(cm)
    # print('Classification Report')
    # print(classification_report(y_val_oh, y_pred_classes_oh))
    # print('Accuracy:', accuracy_score(y_val_oh, y_pred_classes_oh))
    # print('F1 Score:', f1_score(y_val_oh, y_pred_classes_oh, average='weighted'))

    import seaborn as sns
    classes = [str(i) for i in range(5)]
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.grid(True, which='minor', linestyle='-')
    plt.show()


def train(epochs, is_transfer):
    train_ds, val_ds, class_names = data_load("./flower_photos/flower_photos", 224, 224, 4)
    model = model_load((224, 224, 3), is_transfer)
    from datetime import datetime
    from tensorflow.keras.callbacks import TensorBoard

    if is_transfer == 0:
        log_dir = "logs/mobilenet_flower/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    elif is_transfer == 1:
        log_dir = "logs/cnn_flower/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    elif is_transfer == 2:
        log_dir = "logs/restnet_flower/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    #训练模型
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[tensorboard_callback])

    if is_transfer == 0:
        # model.evaluate(val_ds)
        model.save("models/mobilenet_flower.h5")
    elif is_transfer == 1:
        model.save("models/cnn_flower.h5")
    elif is_transfer == 2:
        model.save("models/restnet_flower.h5")
    # train_predictions = model.predict(train_ds, batch_size=10)
    # test_predictions = model.predict(val_ds, batch_size=10)

    show_metrics(history)



if __name__ == '__main__':
    # 创建mobile模型
    # train(epochs=20, is_transfer=0)
    # 创建cnn卷积模型
    # train(epochs=20, is_transfer=1)
    # 创建restnet模型
    train(epochs=20, is_transfer=2)


    # 评估mobile模型
    # evaluate(20, 0)
    # 评估cnn模型
    # evaluate(20, 1)
    # 评估restnet模型
    # evaluate(20, 2)

