import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 查看图片数量
def read_flower_data(folder_name):
    folders = os.listdir(folder_name)
    flower_names = []
    flower_nums = []
    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        images_num = len(images)
        print("{}:{}".format(folder, images_num))
        flower_names.append(folder)
        flower_nums.append(images_num)

    return flower_names, flower_nums


# 绘制柱状图
def show_bar(x, y):
    # 绘图
    plt.barh(range(5), y, align='center', color='steelblue', alpha=0.8)
    # 添加轴标签
    plt.xlabel('num')
    # 添加标题
    plt.title('Num of flowers')
    # 添加刻度标签
    plt.yticks(range(5), x)
    # 设置Y轴的刻度范围
    # plt.xlim([32, 47])
    # 为每个条形图添加数值标签
    for x, y in enumerate(y):
        plt.text(y + 0.1, x, '%s' % y, va='center')
    # 显示图形
    plt.show()


# 绘制不同类别花卉图片的尺寸分布柱状图
def plot_flower_size_distribution(folder_name):
    folders = os.listdir(folder_name)

    width = 0.3

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, folder in enumerate(folders):
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        sizes = [Image.open(os.path.join(folder_path, img)).size for img in images]
        sizes = np.array(sizes)
        sizes = sizes[:, 0]*sizes[:, 1]

        ax.bar(i-width/2, sizes.mean(), width, label=folder+' mean')
        ax.bar(i+width/2, sizes.max(), width, label=folder+' max')

    ax.set_title('Flower Size Distribution')
    ax.set_ylabel('Size')
    ax.set_xticks(range(len(folders)))
    ax.set_xticklabels(folders)
    ax.legend()

    plt.show()

# 绘制不同类别花卉图片的色调分布柱状图
def plot_flower_color_distribution(folder_name):
    folders = os.listdir(folder_name)
    width = 0.3

    colors = ['red', 'green', 'blue', 'gray']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, folder in enumerate(folders):
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        c = np.zeros(4)
        for img in images:
            img_path = os.path.join(folder_path, img)

            img = Image.open(img_path)

            img_array = np.array(img)

            img_red = img_array[:, :, 0]
            img_green = img_array[:, :, 1]
            img_blue = img_array[:, :, 2]
            img_gray = (0.299*img_red + 0.587*img_green + 0.114*img_blue).astype(np.uint8)

            c[0] += img_red.mean()
            c[1] += img_green.mean()
            c[2] += img_blue.mean()
            c[3] += img_gray.mean()

        ax.bar(i*width, c[0]/len(images), width/4, color=colors[0], label=folder+' red')
        ax.bar(i*width+width/4, c[1]/len(images), width/4, color=colors[1], label=folder+' green')
        ax.bar(i*width+2*width/4, c[2]/len(images), width/4, color=colors[2], label=folder+' blue')
        ax.bar(i*width+3*width/4, c[3]/len(images), width/4, color=colors[3], label=folder+' gray')

    ax.set_title('Flower Color Distribution')
    ax.set_ylabel('Color')
    ax.set_xticks(np.arange(len(folders))*width + width/2)
    ax.set_xticklabels(folders)
    ax.legend()

    plt.show()

if __name__ == '__main__':
    src_data_folder = "./flower_photos/flower_photos"
    x, y = read_flower_data(src_data_folder)
    show_bar(x, y)
    plot_flower_size_distribution(src_data_folder)
    plot_flower_color_distribution(src_data_folder)
