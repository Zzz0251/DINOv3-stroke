# RGB

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def resize_with_binary_labels(image_path, target_size=(224, 224)):
    """
    使用最近邻插值处理RGB图像，并将G通道二值化：>127的变成255，<=127的变成0
    """
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    target_h, target_w = target_size
    
    # 分离通道
    r_channel = img[:, :, 0]  # 原图
    g_channel = img[:, :, 1]  # 标签
    b_channel = img[:, :, 2]  # 通常是0
    
    # R通道使用最近邻插值
    r_resized = cv2.resize(r_channel, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # G通道先resize再二值化
    g_resized = cv2.resize(g_channel, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    # 以127为分界：>127的变成255，<=127的变成0
    g_binary = (g_resized > 127).astype(np.uint8) * 255
    
    # B通道使用最近邻插值
    b_resized = cv2.resize(b_channel, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # 重新组合
    result = np.stack([r_resized, g_binary, b_resized], axis=2)
    
    return result

def process_dataset(dataset_path, output_path, target_size=(224, 224)):
    """
    处理整个数据集
    """
    os.makedirs(output_path, exist_ok=True)
    
    processed_count = 0
    
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(dataset_path, filename)
            
            try:
                # 使用最近邻插值resize并二值化标签
                resized_img = resize_with_binary_labels(input_path, target_size)
                
                # 保存结果
                output_file = os.path.join(output_path, filename)
                cv2.imwrite(output_file, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed: {processed_count} files")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Total processed: {processed_count} files")

def visualize_before_after(image_path, target_size=(224, 224)):
    """
    可视化resize前后的对比
    """
    # 原图
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Resize后
    resized = resize_with_binary_labels(image_path, target_size)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 原图RGB
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f'Original RGB ({original.shape[1]}x{original.shape[0]})')
    axes[0, 0].axis('off')
    
    # 原图标签
    axes[1, 0].imshow(original[:, :, 1], cmap='gray')
    axes[1, 0].set_title('Original Label (G channel)')
    axes[1, 0].axis('off')
    
    # Resize后RGB
    axes[0, 1].imshow(resized)
    axes[0, 1].set_title(f'Resized RGB ({resized.shape[1]}x{resized.shape[0]})')
    axes[0, 1].axis('off')
    
    # Resize后标签
    axes[1, 1].imshow(resized[:, :, 1], cmap='gray')
    axes[1, 1].set_title('Resized Binary Label (threshold=127)')
    axes[1, 1].axis('off')
    
    # 检查标签的唯一值和像素数量
    original_unique = np.unique(original[:, :, 1])
    resized_unique = np.unique(resized[:, :, 1])
    
    original_fg_count = np.sum(original[:, :, 1] > 127)
    resized_fg_count = np.sum(resized[:, :, 1] > 127)
    
    print(f"Original label unique values: {original_unique}")
    print(f"Resized label unique values: {resized_unique}")
    print(f"Original foreground pixels (>127): {original_fg_count}")
    print(f"Resized foreground pixels (>127): {resized_fg_count}")
    
    # 显示阈值化统计
    original_above_127 = original_unique[original_unique > 127]
    print(f"Original values that become foreground (>127): {original_above_127}")
    
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    dataset_path = "/home/donghao/dinov3/Dataset106_infarct_isles24/slices_10_test"
    output_path = "/home/donghao/dinov3/Dataset106_infarct_isles24/slices_10_test_224"
    
    # 先看一个样本的效果
    sample_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if sample_files:
        sample_path = os.path.join(dataset_path, sample_files[0])
        print(f"Visualizing sample: {sample_files[0]}")
        visualize_before_after(sample_path, target_size=(224, 224))
    
    # 处理整个数据集
    print("\nProcessing entire dataset...")
    process_dataset(dataset_path, output_path, target_size=(224, 224))
    print("Done!")