import matplotlib.pyplot as plt
import numpy as np



def show_three_images(im1, im2, im3, img_name):

    plt.figure(figsize=(15, 5))  # 宽度15，高度5，可根据需要调整

    # 第一张图（子图1）
    plt.subplot(1, 3, 1)  # 1行3列，第1个位置
    plt.imshow(im1, cmap='gray')
    plt.title(f'title 1')  # 替换为你的标题

    # 第二张图（子图2）
    plt.subplot(1, 3, 2)  # 1行3列，第2个位置
    plt.imshow(im2, cmap='gray')
    plt.title(f'title 2')

    # 第三张图（子图3）
    plt.subplot(1, 3, 3)  # 1行3列，第3个位置
    plt.imshow(im3, cmap='gray')
    plt.title(f'title 3')

    # 调整子图之间的间距，避免重叠
    plt.tight_layout()

    # 保存整个窗口（包含3张图）
    plt.savefig(f"{img_name}_three_images.png")  # 保存为整体图片

    # 显示窗口
    plt.show()
    
def show_one_images(im1, img_name):

    plt.figure(figsize=(10, 10))  # 宽度10，高度10，可根据需要调整

    # 第一张图（子图1）
    plt.imshow(im1, cmap='gray')
    plt.title(f'title 1')  # 替换为你的标题

    
    # 保存整个窗口（包含3张图）
    plt.savefig(f"{img_name}_one_images.png")  # 保存为整体图片

    # 显示窗口
    plt.show()