# normal hemo classify HeadCT dataset


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
import seaborn as sns

class ClassificationDataset(Dataset):
    def __init__(self, data_dir, labels_csv, processor, is_train=True):
        self.data_dir = data_dir
        self.processor = processor
        self.is_train = is_train
        
        # 读取标签文件
        self.labels_df = pd.read_csv(labels_csv)
        
        # 获取数据目录中的图像文件
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
        # 匹配图像文件和标签
        self.data = []
        for img_file in image_files:
            img_id = int(img_file.split('.')[0])  # 从文件名提取ID
            label_row = self.labels_df[self.labels_df['id'] == img_id]
            if not label_row.empty:
                label = label_row.iloc[0]['label']
                self.data.append((img_file, label))
        
        print(f"Found {len(self.data)} samples in {data_dir}")
        
        # 数据增强（仅训练时）
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224))
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_file, label = self.data[idx]
        img_path = os.path.join(self.data_dir, img_file)
        
        # 加载灰度图像并转换为RGB（DINOv3需要3通道）
        image = Image.open(img_path).convert('L')  # 确保是灰度图
        image = image.convert('RGB')  # 转换为RGB
        
        # 应用变换
        image = self.transform(image)
        
        # 使用processor处理图像
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # 移除batch维度
        
        return pixel_values, torch.tensor(label, dtype=torch.long)

class DINOv3ClassificationModel(nn.Module):
    def __init__(self, dinov3_model_path, num_classes=2):
        super().__init__()
        
        # 加载DINOv3模型
        self.dinov3 = AutoModel.from_pretrained(dinov3_model_path)
        
        # 冻结DINOv3参数（linear probe设置）
        for param in self.dinov3.parameters():
            param.requires_grad = False
            
        # 获取特征维度
        self.feature_dim = self.dinov3.config.hidden_size
        
        # 检测模型类型
        model_name = os.path.basename(dinov3_model_path).lower()
        if 'convnext' in model_name:
            self.model_type = 'convnext'
            print(f"Detected ConvNeXt model. Feature dim: {self.feature_dim}")
        else:
            self.model_type = 'vit'
            # DINOv3 ViT的架构信息
            self.patch_size = 16
            self.num_class_tokens = 1
            self.num_register_tokens = 4
            print(f"Detected ViT model. Feature dim: {self.feature_dim}")
        
        # Linear probe分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        # 获取DINOv3特征
        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)
        
        if self.model_type == 'convnext':
            # ConvNeXt: 使用全局平均池化
            last_hidden_state = outputs.last_hidden_state  # [B, C, H, W]
            features = torch.mean(last_hidden_state, dim=(2, 3))  # 全局平均池化 [B, C]
        else:
            # ViT: 使用class token
            last_hidden_state = outputs.last_hidden_state  # [B, N, D]
            features = last_hidden_state[:, 0, :]  # 取class token [B, D]
        
        # 通过分类头
        logits = self.classifier(features)
        
        return logits

def calculate_metrics(y_true, y_pred, y_prob):
    """计算分类指标"""
    # 准确率
    acc = accuracy_score(y_true, y_pred)
    
    # AUC
    auc = roc_auc_score(y_true, y_prob[:, 1])
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 敏感性（召回率）
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 特异性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return acc, auc, sensitivity, specificity, cm

def create_output_dir(model_path):
    """创建输出目录"""
    model_name = os.path.basename(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_classification/{model_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    return output_dir

def train_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据路径
    train_dir = "/datapath/to/train"
    test_dir = "/datapath/to/test"
    labels_csv = "/datapath/to/labels.csv"
    
    # 模型路径
    model_path = "/path/to/weights/dinov3-vith16plus-pretrain-lvd1689m"
    
    # 创建输出目录
    output_dir = create_output_dir(model_path)
    print(f"Output directory: {output_dir}")
    
    # 加载处理器
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # 创建数据集和数据加载器
    train_dataset = ClassificationDataset(train_dir, labels_csv, processor, is_train=True)
    test_dataset = ClassificationDataset(test_dir, labels_csv, processor, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 检查标签分布
    train_labels = [label for _, label in train_dataset.data]
    test_labels = [label for _, label in test_dataset.data]
    print(f"Train label distribution - Normal: {train_labels.count(0)}, Hemorrhage: {train_labels.count(1)}")
    print(f"Test label distribution - Normal: {test_labels.count(0)}, Hemorrhage: {test_labels.count(1)}")
    
    # 创建模型
    model = DINOv3ClassificationModel(model_path, num_classes=2).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # 训练历史记录
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # 训练循环
    num_epochs = 50
    best_auc = 0
    
    # 保存训练配置
    config_info = f"""Training Configuration:
Model: {os.path.basename(model_path)}
Task: Binary Classification (Normal vs Hemorrhage)
Device: {device}
Epochs: {num_epochs}
Batch Size: 32
Learning Rate: 1e-3
Weight Decay: 1e-4
Train Samples: {len(train_dataset)}
Test Samples: {len(test_dataset)}
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
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
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
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算指标
        all_probabilities = np.array(all_probabilities)
        acc, auc, sen, spe, cm = calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # 计算平均值
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(test_loader)
        
        # 保存训练历史
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_acc'].append(avg_train_acc)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_acc'].append(acc)
        train_history['val_auc'].append(auc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}, Val AUC: {auc:.4f}')
        print(f'Sensitivity: {sen:.4f}, Specificity: {spe:.4f}')
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型（基于AUC）
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            print(f'New best model saved with AUC: {best_auc:.4f}')
        
        print('-' * 50)
    
    # 保存训练历史
    import json
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_history['epoch'], train_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(train_history['epoch'], train_history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_history['epoch'], train_history['train_acc'], 'b-', label='Train Acc')
    plt.plot(train_history['epoch'], train_history['val_acc'], 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_history['epoch'], train_history['val_auc'], 'g-', label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir

def evaluate_model(output_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据路径
    test_dir = "/path/to/test"
    labels_csv = "/path/to/labels.csv"
    
    # 加载处理器和测试数据
    processor = AutoImageProcessor.from_pretrained(model_path)
    test_dataset = ClassificationDataset(test_dir, labels_csv, processor, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载模型
    model = DINOv3ClassificationModel(model_path, num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoints', 'best_model.pth')))
    model.eval()
    
    # 评估
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算最终指标
    all_probabilities = np.array(all_probabilities)
    acc, auc, sen, spe, cm = calculate_metrics(all_labels, all_predictions, all_probabilities)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Hemorrhage'], 
                yticklabels=['Normal', 'Hemorrhage'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    results_text = f"""Final Evaluation Results:
Model: {os.path.basename(model_path)}
Evaluation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test Samples: {len(test_dataset)}

Classification Metrics:
Accuracy (ACC): {acc:.4f} ({acc*100:.2f}%)
AUC: {auc:.4f}
Sensitivity (SEN): {sen:.4f} ({sen*100:.2f}%)
Specificity (SPE): {spe:.4f} ({spe*100:.2f}%)

Confusion Matrix:
True Negative (TN): {cm[0,0]}
False Positive (FP): {cm[0,1]}
False Negative (FN): {cm[1,0]}
True Positive (TP): {cm[1,1]}

Notes:
- Accuracy: Overall correct predictions
- AUC: Area Under the ROC Curve
- Sensitivity: True positive rate (TP/(TP+FN))
- Specificity: True negative rate (TN/(TN+FP))
"""
    
    print(results_text)
    
    # 保存结果到文件
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(results_text)
    
    return acc, auc, sen, spe

if __name__ == "__main__":
    print("Starting classification training...")
    output_dir = train_model()
    
    print(f"\nStarting evaluation...")
    model_path = "/path/to/weights/dinov3-vith16plus-pretrain-lvd1689m"
    evaluate_model(output_dir, model_path)
    
    print(f"\nAll results saved to: {output_dir}")