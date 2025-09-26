import os
import shutil
import random
import json
from pathlib import Path
from PIL import Image
import numpy as np

def crop_to_square(image, target_size=512):
    """
    将图像crop到正方形，如果不足target_size则按短边crop
    """
    width, height = image.size
    
    # 如果图像小于目标尺寸，按短边crop
    if min(width, height) < target_size:
        crop_size = min(width, height)
    else:
        crop_size = target_size
    
    # 计算crop的起始位置（中心crop）
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    # 执行crop
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

def process_and_save_image(input_path, output_path, target_crop_size=512, target_resize_size=256):
    """
    处理单张图像：crop到正方形然后resize
    """
    try:
        # 打开图像
        with Image.open(input_path) as img:
            # Crop到正方形
            cropped_img = crop_to_square(img, target_crop_size)
            
            # Resize到目标尺寸
            resized_img = cropped_img.resize((target_resize_size, target_resize_size), Image.LANCZOS)
            
            # 保存图像
            resized_img.save(output_path)
            
        return True
    except Exception as e:
        print(f"处理图像失败 {input_path}: {str(e)}")
        return False

def process_dataset(source_dir, processed_dir, target_crop_size=512, target_resize_size=256):
    """
    处理整个数据集
    """
    # 创建处理后的目录
    os.makedirs(processed_dir, exist_ok=True)
    
    # 获取所有图像文件
    source_path = Path(source_dir)
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"开始处理图像: crop到{target_crop_size}x{target_crop_size}, resize到{target_resize_size}x{target_resize_size}")
    
    success_count = 0
    
    for i, file_path in enumerate(image_files):
        output_path = Path(processed_dir) / file_path.name
        
        if process_and_save_image(file_path, output_path, target_crop_size, target_resize_size):
            success_count += 1
        
        # 显示进度
        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"已处理: {i + 1}/{len(image_files)} ({(i + 1)/len(image_files)*100:.1f}%)")
    
    print(f"图像处理完成! 成功处理 {success_count}/{len(image_files)} 个文件")
    return success_count

def split_dataset(source_dir, train_dir, test_dir, train_ratio=0.8, random_seed=42):
    """
    将数据集随机分割为训练集和测试集
    """
    # 设置随机种子确保可重现性
    random.seed(random_seed)
    
    # 创建输出目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 获取所有文件
    source_path = Path(source_dir)
    all_files = [f for f in source_path.iterdir() 
                 if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    print(f"准备分割 {len(all_files)} 个文件")
    
    # 随机打乱文件列表
    random.shuffle(all_files)
    
    # 计算分割点
    train_count = int(len(all_files) * train_ratio)
    test_count = len(all_files) - train_count
    
    # 分割文件列表
    train_files = all_files[:train_count]
    test_files = all_files[train_count:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")
    
    # 复制训练集文件
    print("正在复制训练集文件...")
    for file_path in train_files:
        dest_path = Path(train_dir) / file_path.name
        shutil.copy2(file_path, dest_path)
    
    # 复制测试集文件
    print("正在复制测试集文件...")
    for file_path in test_files:
        dest_path = Path(test_dir) / file_path.name
        shutil.copy2(file_path, dest_path)
    
    # 准备splits信息
    splits = {
        'train': [f.name for f in train_files],
        'test': [f.name for f in test_files],
        'train_count': len(train_files),
        'test_count': len(test_files),
        'total_count': len(all_files),
        'train_ratio': train_ratio,
        'random_seed': random_seed
    }
    
    return splits

def main():
    # 定义路径
    source_dir = "/home/donghao/dinov3/Dataset108_hemo_BHSD/slices_10"
    processed_dir = "/home/donghao/dinov3/Dataset108_hemo_BHSD/slices_10_processed"
    train_dir = "/home/donghao/dinov3/Dataset108_hemo_BHSD/slices_10_train"
    test_dir = "/home/donghao/dinov3/Dataset108_hemo_BHSD/slices_10_test"
    splits_file = "/home/donghao/dinov3/Dataset108_hemo_BHSD/slices_10/splits_10.json"
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在 {source_dir}")
        return
    
    print("=" * 60)
    print("步骤1: 图像预处理 (crop + resize)")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"处理后目录: {processed_dir}")
    
    # 步骤1: 处理图像 (crop到512x512, resize到256x256)
    success_count = process_dataset(source_dir, processed_dir, 
                                  target_crop_size=512, 
                                  target_resize_size=224)
    
    if success_count == 0:
        print("图像处理失败，停止执行")
        return
    
    print("\n" + "=" * 60)
    print("步骤2: 数据集分割 (8:2)")
    print("=" * 60)
    print(f"处理后目录: {processed_dir}")
    print(f"训练集目录: {train_dir}")
    print(f"测试集目录: {test_dir}")
    
    # 步骤2: 分割数据集
    splits = split_dataset(processed_dir, train_dir, test_dir, 
                          train_ratio=0.8, random_seed=42)
    
    # 保存splits信息到JSON文件
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"原始图像: {len(os.listdir(source_dir))} 个文件")
    print(f"处理成功: {success_count} 个文件")
    print(f"训练集: {splits['train_count']} 个文件")
    print(f"测试集: {splits['test_count']} 个文件")
    print(f"图像尺寸: 256x256")
    print(f"splits信息已保存到: {splits_file}")
    
    # 显示一些样例文件名
    print("\n训练集样例文件:")
    for i, filename in enumerate(splits['train'][:3]):
        print(f"  {filename}")
    if len(splits['train']) > 3:
        print(f"  ... 还有 {len(splits['train']) - 3} 个文件")
    
    print("\n测试集样例文件:")
    for i, filename in enumerate(splits['test'][:3]):
        print(f"  {filename}")
    if len(splits['test']) > 3:
        print(f"  ... 还有 {len(splits['test']) - 3} 个文件")

if __name__ == "__main__":
    main()