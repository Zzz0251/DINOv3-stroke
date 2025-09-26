# change files name and generate csv label
# 生成的csv文件，第一列是id第二列是label，0表示normal 1表示hemo 2表示infarct

import os
import pandas as pd
import shutil
from pathlib import Path

def preprocess_dataset():
    # 数据集路径
    base_path = "/path/to/Dataset102_3classes"
    categories = {
        'hemo': os.path.join(base_path, 'hemo'),
        'normal': os.path.join(base_path, 'normal'), 
        'infarct': os.path.join(base_path, 'infarct')
    }
    
    # 标签映射
    label_mapping = {
        'normal': 0,
        'hemo': 1,
        'infarct': 2
    }
    
    # 存储所有文件信息
    all_files = []
    
    # 处理每个类别
    for category, folder_path in categories.items():
        print(f"处理类别: {category}")
        
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            continue
        
        # 获取所有PNG文件
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        print(f"找到 {len(png_files)} 个PNG文件")
        
        # 重命名文件
        for i, old_filename in enumerate(png_files, 1):
            # 生成新文件名
            new_filename = f"{category}_{i:06d}.png"
            
            old_filepath = os.path.join(folder_path, old_filename)
            new_filepath = os.path.join(folder_path, new_filename)
            
            try:
                # 重命名文件
                os.rename(old_filepath, new_filepath)
                
                # 记录文件信息 (去掉.png扩展名作为id)
                file_id = os.path.splitext(new_filename)[0]
                all_files.append({
                    'id': file_id,
                    'label': label_mapping[category]
                })
                
                if i % 500 == 0:
                    print(f"  已处理 {i}/{len(png_files)} 个文件")
                    
            except Exception as e:
                print(f"重命名文件失败 {old_filename}: {e}")
    
    # 生成CSV文件
    if all_files:
        df = pd.DataFrame(all_files)
        csv_path = os.path.join(base_path, 'labels.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\n处理完成!")
        print(f"总共处理了 {len(all_files)} 个文件")
        print(f"CSV文件已保存到: {csv_path}")
        
        # 打印统计信息
        print(f"\n各类别文件数量:")
        for category, label in label_mapping.items():
            count = df[df['label'] == label].shape[0]
            print(f"  {category} (label={label}): {count} 个文件")
        
        # 显示CSV前几行
        print(f"\nCSV文件预览:")
        print(df.head(10))
    
    else:
        print("没有处理任何文件")

# 运行
if __name__ == "__main__":
    preprocess_dataset()
