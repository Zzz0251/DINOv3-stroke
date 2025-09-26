# 3D classification with DINOv3 - Balanced & AUC-optimized (Fixed)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import nibabel as nib
import math
import json

class Classification3DDataset(Dataset):
    def __init__(self, data_list, data_dir, processor, is_train=True, num_slices=16):
        """
        data_list: list of tuples (filename, label)
        """
        self.data_dir = data_dir
        self.processor = processor
        self.is_train = is_train
        self.num_slices = num_slices
        self.data = data_list
        
        print(f"Dataset created with {len(self.data)} samples")
        
        # 检查标签分布
        labels = [label for _, label in self.data]
        unique_labels = list(set(labels))
        print(f"Label distribution:")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        # 数据增强（仅训练时）
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.data)
    
    def normalize_to_0_255(self, volume):
        """将volume标准化到0-255范围"""
        volume = volume.astype(np.float32)
        volume_min = volume.min()
        volume_max = volume.max()
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min) * 255
        else:
            volume = np.zeros_like(volume)
        return volume.astype(np.uint8)
    
    def resize_volume(self, volume, target_shape=(224, 224)):
        """将每个切片resize到目标尺寸"""
        resized_volume = np.zeros((volume.shape[0], target_shape[0], target_shape[1]))
        for i in range(volume.shape[0]):
            slice_img = Image.fromarray(volume[i])
            slice_img = slice_img.resize(target_shape, Image.BILINEAR)
            resized_volume[i] = np.array(slice_img)
        return resized_volume
    
    def sample_slices(self, volume):
        """从3D volume中采样固定数量的切片"""
        depth = volume.shape[0]
        
        if depth <= self.num_slices:
            indices = np.linspace(0, depth-1, self.num_slices, dtype=int)
        else:
            indices = np.linspace(0, depth-1, self.num_slices, dtype=int)
        
        sampled_volume = volume[indices]
        return sampled_volume
    
    def __getitem__(self, idx):
        nii_file, label = self.data[idx]
        nii_path = os.path.join(self.data_dir, nii_file)
        
        try:
            # 加载nii文件
            nii_data = nib.load(nii_path)
            volume = nii_data.get_fdata()
            
            # 转换维度顺序为 [D, H, W]
            volume = np.transpose(volume, (2, 0, 1))
            
        except Exception as e:
            print(f"Error loading {nii_file}: {e}")
            volume = np.zeros((self.num_slices, 224, 224))
        
        # 标准化和采样
        volume = self.normalize_to_0_255(volume)
        volume = self.sample_slices(volume)
        volume = self.resize_volume(volume, (224, 224))
        
        # 为每个切片准备DINOv3输入
        slice_features = []
        for i in range(self.num_slices):
            slice_img = Image.fromarray(volume[i]).convert('RGB')
            
            # 数据增强
            if self.transform and self.is_train:
                slice_img = self.transform(slice_img)
            
            inputs = self.processor(images=slice_img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            slice_features.append(pixel_values)
        
        volume_features = torch.stack(slice_features)  # [num_slices, 3, 224, 224]
        
        return volume_features, torch.tensor(label, dtype=torch.long)

class DINOv3Classification3DModel(nn.Module):
    def __init__(self, dinov3_model_path, num_classes=2, num_slices=16, aggregation='mean'):
        super().__init__()
        
        self.num_slices = num_slices
        self.aggregation = aggregation
        
        # 加载DINOv3模型
        self.dinov3 = AutoModel.from_pretrained(dinov3_model_path)
        
        # 冻结DINOv3参数
        for param in self.dinov3.parameters():
            param.requires_grad = False
            
        self.feature_dim = self.dinov3.config.hidden_size
        
        # 检测模型类型
        model_name = os.path.basename(dinov3_model_path).lower()
        if 'convnext' in model_name:
            self.model_type = 'convnext'
            print(f"Detected ConvNeXt model. Feature dim: {self.feature_dim}")
        else:
            self.model_type = 'vit'
            self.patch_size = 16
            self.num_class_tokens = 1
            self.num_register_tokens = 4
            print(f"Detected ViT model. Feature dim: {self.feature_dim}")
        
        # 根据聚合方式设置分类头
        if aggregation == 'concat':
            input_dim = self.feature_dim * num_slices
        elif aggregation in ['mean', 'max']:
            input_dim = self.feature_dim
        elif aggregation == 'attention':
            input_dim = self.feature_dim
            # 注意力权重网络
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        print(f"Using {aggregation} aggregation. Input dim to classifier: {input_dim}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def extract_2d_features(self, x):
        """从2D切片批量提取DINOv3特征"""
        B, C, H, W = x.shape
        
        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)
        
        if self.model_type == 'convnext':
            # ConvNeXt: 使用全局平均池化
            last_hidden_state = outputs.last_hidden_state  # [B, C, H, W]
            features = torch.mean(last_hidden_state, dim=(2, 3))  # [B, C]
        else:
            # ViT: 使用class token
            last_hidden_state = outputs.last_hidden_state  # [B, N, D]
            features = last_hidden_state[:, 0, :]  # 取class token [B, D]
        
        return features
    
    def forward(self, x):
        # x shape: [B, num_slices, 3, 224, 224]
        B, D, C, H, W = x.shape
        
        # 重塑为 [B*D, 3, 224, 224] 以便批量处理所有切片
        x_reshaped = x.view(B * D, C, H, W)
        
        # 批量提取所有切片的DINOv3特征
        slice_features = self.extract_2d_features(x_reshaped)  # [B*D, feature_dim]
        
        # 重塑回 [B, D, feature_dim]
        slice_features = slice_features.view(B, D, self.feature_dim)
        
        # 根据聚合方式处理特征
        if self.aggregation == 'concat':
            # 拼接所有切片特征
            volume_features = slice_features.reshape(B, -1)  # [B, D * feature_dim]
        
        elif self.aggregation == 'mean':
            # 平均池化
            volume_features = torch.mean(slice_features, dim=1)  # [B, feature_dim]
        
        elif self.aggregation == 'max':
            # 最大池化
            volume_features, _ = torch.max(slice_features, dim=1)  # [B, feature_dim]
        
        elif self.aggregation == 'attention':
            # 注意力加权平均
            attention_weights = self.attention(slice_features)  # [B, D, 1]
            attention_weights = F.softmax(attention_weights, dim=1)  # [B, D, 1]
            volume_features = torch.sum(slice_features * attention_weights, dim=1)  # [B, feature_dim]
        
        # 通过分类头
        logits = self.classifier(volume_features)
        
        return logits

def find_optimal_threshold(y_true, y_prob):
    """根据ROC曲线找到最优阈值（离左上角最近的点）"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # 计算每个点到左上角(0,1)的距离
    distances = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    optimal_idx = np.argmin(distances)
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    # 确保阈值在合理范围内
    if optimal_threshold == np.inf:
        optimal_threshold = 1.0
    elif optimal_threshold == -np.inf:
        optimal_threshold = 0.0
    
    return float(optimal_threshold), float(optimal_fpr), float(optimal_tpr)

def calculate_metrics_with_optimal_threshold(y_true, y_pred, y_prob):
    """使用最优阈值计算分类指标"""
    # 准确率（基于argmax预测）
    acc = accuracy_score(y_true, y_pred)
    
    # AUC（只在二分类时计算）
    if len(np.unique(y_true)) == 2 and y_prob.shape[1] == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        # 找到最优阈值
        optimal_threshold, optimal_fpr, optimal_tpr = find_optimal_threshold(y_true, y_prob[:, 1])
        
        # 使用最优阈值重新预测
        y_pred_optimal = (y_prob[:, 1] >= optimal_threshold).astype(int)
        
        # 计算混淆矩阵（基于最优阈值）
        cm = confusion_matrix(y_true, y_pred_optimal)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        else:
            sensitivity = 0.0
            specificity = 0.0
        
        return acc, auc, sensitivity, specificity, cm, optimal_threshold, optimal_fpr, optimal_tpr
    else:
        # 多分类情况
        cm = confusion_matrix(y_true, y_pred)
        return acc, 0.0, 0.0, 0.0, cm, 0.5, 0.0, 0.0

def create_stratified_split(data_list, test_size=0.2, random_state=42):
    """创建分层采样的训练/测试集"""
    files = [item[0] for item in data_list]
    labels = [item[1] for item in data_list]
    
    # 分层采样
    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=test_size, random_state=random_state, 
        stratify=labels
    )
    
    train_data = list(zip(train_files, train_labels))
    test_data = list(zip(test_files, test_labels))
    
    return train_data, test_data

def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def create_output_dir(model_path):
    """创建输出目录"""
    model_name = os.path.basename(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_lesionage_att/{model_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    return output_dir

def train_model_3d_classification():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    data_dir = "/path/to/3d_nifti_dataset"
    labels_csv = "/path/to/labels.csv"


    # 模型路径
    model_path = "/path/to/weights/dinov3-vitb16-pretrain-lvd1689m"
    
    # 创建输出目录
    output_dir = create_output_dir(model_path + "_3D_Classification_Balanced")
    print(f"Output directory: {output_dir}")
    
    # 加载处理器
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # 读取标签文件
    labels_df = pd.read_csv(labels_csv)
    print(f"Labels CSV shape: {labels_df.shape}")
    print(f"Labels columns: {list(labels_df.columns)}")
    print(f"First few rows:\n{labels_df.head()}")
    
    # 获取数据目录中的nii.gz文件
    nii_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    print(f"Found {len(nii_files)} .nii.gz files in {data_dir}")
    
    # nii_files = [f for f in os.listdir(data_dir) if f.endswith('.nii')]
    # print(f"Found {len(nii_files)} .nii files in {data_dir}")
    # 匹配文件和标签
    all_data = []
    for nii_file in nii_files:
        # 从文件名提取ID
        file_id = nii_file.replace('.nii.gz', '')
        # file_id = nii_file.replace('.nii', '')
        if '_' in file_id:
            file_id = file_id.split('_')[0]
        
        # 在标签文件中查找对应的标签
        label_row = labels_df[labels_df.iloc[:, 0] == file_id]
        if not label_row.empty:
            label = int(label_row.iloc[0, 1])
            all_data.append((nii_file, label))
            if len(all_data) <= 5:
                print(f"Matched: {nii_file} -> ID: {file_id} -> Label: {label}")
    
    print(f"Successfully matched {len(all_data)} samples")
    
    if len(all_data) == 0:
        print("Error: No valid data found!")
        return None
    
    # 分层采样划分训练集和测试集
    train_data, test_data = create_stratified_split(all_data, test_size=0.2, random_state=42)
    
    print(f"Stratified split results:")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # 检查分层后的标签分布
    train_labels = [label for _, label in train_data]
    test_labels = [label for _, label in test_data]
    
    print(f"Train label distribution:")
    for label in set(train_labels):
        count = train_labels.count(label)
        print(f"  Label {label}: {count} samples ({count/len(train_labels)*100:.1f}%)")
    
    print(f"Test label distribution:")
    for label in set(test_labels):
        count = test_labels.count(label)
        print(f"  Label {label}: {count} samples ({count/len(test_labels)*100:.1f}%)")
    
    # 创建数据集
    num_slices = 16
    aggregation = 'attention'  #concat max mean attention
    
    train_dataset = Classification3DDataset(train_data, data_dir, processor, is_train=True, num_slices=num_slices)
    test_dataset = Classification3DDataset(test_data, data_dir, processor, is_train=False, num_slices=num_slices)
    
    # 计算类别权重来处理不平衡
    all_labels = [label for _, label in all_data]
    num_classes = len(set(all_labels))
    class_counts = [all_labels.count(i) for i in range(num_classes)]
    total_samples = len(all_labels)
    class_weights = [total_samples / (num_classes * count) for count in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # 创建模型
    model = DINOv3Classification3DModel(
        model_path, 
        num_classes=num_classes, 
        num_slices=num_slices,
        aggregation=aggregation
    ).to(device)
    
    # 损失函数和优化器（使用类别权重）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)  # 基于AUC最大化
    
    # 训练历史记录
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'optimal_threshold': []
    }
    
    # 训练循环
    num_epochs = 50
    best_auc = 0
    debug_printed = False
    
    # 保存训练配置
    config_info = f"""Training Configuration (3D Classification - Balanced):
Model: {os.path.basename(model_path)}
Task: 3D Classification with Stratified Split
Aggregation: {aggregation}
Device: {device}
Epochs: {num_epochs}
Batch Size: 4
Learning Rate: 1e-3
Weight Decay: 1e-4
Number of Slices: {num_slices}
Number of Classes: {num_classes}
Class Weights: {class_weights.cpu().numpy()}
Total Samples: {len(all_data)}
Train Samples: {len(train_data)}
Test Samples: {len(test_data)}
Output Directory: {output_dir}
Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(config_info)
    
    print(config_info)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for volumes, labels in pbar:
            volumes, labels = volumes.to(device), labels.to(device)
            
            if not debug_printed:
                print(f"Volumes shape: {volumes.shape}")
                print(f"Labels shape: {labels.shape}")
                outputs = model(volumes)
                print(f"Model output shape: {outputs.shape}")
                debug_printed = True
                print(f"3D Classification with {aggregation} aggregation debugging completed. Continuing training...")
            
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for volumes, labels in test_loader:
                volumes, labels = volumes.to(device), labels.to(device)
                outputs = model(volumes)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算指标（使用最优阈值）
        all_probabilities = np.array(all_probabilities)
        metrics = calculate_metrics_with_optimal_threshold(all_labels, all_predictions, all_probabilities)
        acc, auc, sen, spe, cm, optimal_threshold, optimal_fpr, optimal_tpr = metrics
        
        # 计算平均值
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(test_loader)
        
        # 保存训练历史
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(float(avg_train_loss))
        train_history['train_acc'].append(float(avg_train_acc))
        train_history['val_loss'].append(float(avg_val_loss))
        train_history['val_acc'].append(float(acc))
        train_history['val_auc'].append(float(auc))
        train_history['val_sensitivity'].append(float(sen))
        train_history['val_specificity'].append(float(spe))
        train_history['optimal_threshold'].append(float(optimal_threshold))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}')
        if auc > 0:
            print(f'Val AUC: {auc:.4f} (Best: {best_auc:.4f})')
            print(f'Optimal Threshold: {optimal_threshold:.4f}')
            print(f'Sensitivity: {sen:.4f}, Specificity: {spe:.4f}')
            print(f'Optimal Point: FPR={optimal_fpr:.4f}, TPR={optimal_tpr:.4f}')
        
        # 学习率调度（基于AUC）
        scheduler.step(auc if auc > 0 else acc)
        
        # 保存最佳模型（基于AUC）
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            
            # 保存最佳模型的详细结果
            best_results = {
                'epoch': epoch + 1,
                'auc': float(auc),
                'accuracy': float(acc),
                'sensitivity': float(sen),
                'specificity': float(spe),
                'optimal_threshold': float(optimal_threshold),
                'optimal_fpr': float(optimal_fpr),
                'optimal_tpr': float(optimal_tpr),
                'confusion_matrix': convert_to_serializable(cm)
            }
            
            with open(os.path.join(output_dir, 'best_results.json'), 'w') as f:
                json.dump(best_results, f, indent=2)
            
            print(f'New best model saved with AUC: {best_auc:.4f}')
        
        print('-' * 50)
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    
    # 绘制训练曲线
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 4, 1)
    plt.plot(train_history['epoch'], train_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(train_history['epoch'], train_history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 2)
    plt.plot(train_history['epoch'], train_history['train_acc'], 'b-', label='Train Acc')
    plt.plot(train_history['epoch'], train_history['val_acc'], 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 3)
    plt.plot(train_history['epoch'], train_history['val_auc'], 'g-', label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 4)
    plt.plot(train_history['epoch'], train_history['val_sensitivity'], 'orange', label='Sensitivity')
    plt.plot(train_history['epoch'], train_history['val_specificity'], 'purple', label='Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Sensitivity and Specificity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 4, 5)
    plt.plot(train_history['epoch'], train_history['optimal_threshold'], 'brown', label='Optimal Threshold')
    plt.xlabel('Epoch')
    plt.ylabel('Threshold')
    plt.title('Optimal Threshold')
    plt.legend()
    plt.grid(True)
    
    # 绘制最终的ROC曲线
    plt.subplot(2, 4, 6)
    if num_classes == 2 and len(all_labels) > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([optimal_fpr], [optimal_tpr], 'ro', markersize=8, label=f'Optimal Point')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
    
    # 绘制混淆矩阵
    plt.subplot(2, 4, 7)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Optimal Threshold)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_3d_classification_balanced_{aggregation}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir

if __name__ == "__main__":
    print("Starting 3D classification training with DINOv3 (Balanced & AUC-optimized)...")
    output_dir = train_model_3d_classification()
    if output_dir:
        print(f"\nAll results saved to: {output_dir}")
    else:
        print("Training failed due to data loading issues.")