# normal hemo infarct 3 classes classify our own dataset


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import json
import random

def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_train_test_split(base_path, test_size=0.2, random_state=42):
    """创建训练测试数据划分"""
    set_seed(random_state)
    
    categories = ['normal', 'hemo', 'infarct']
    all_data = []
    
    # 收集所有数据
    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.png')]
            for file in files:
                all_data.append({
                    'filename': file,
                    'category': category,
                    'full_path': os.path.join(category_path, file)
                })
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 标签映射
    label_mapping = {'normal': 0, 'hemo': 1, 'infarct': 2}
    df['label'] = df['category'].map(label_mapping)
    
    # 分层划分训练测试集
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    print(f"Dataset split (random_state={random_state}):")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # 打印各类别分布
    for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
        print(f"\n{split_name} distribution:")
        for category in categories:
            count = len(split_df[split_df['category'] == category])
            percentage = count / len(split_df) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
    
    return train_df, test_df

class ThreeClassDataset(Dataset):
    def __init__(self, dataframe, processor, is_train=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.is_train = is_train
        
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
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['full_path']
        label = row['label']
        
        # 加载灰度图像并转换为RGB
        image = Image.open(img_path).convert('L')
        image = image.convert('RGB')
        
        # 应用变换
        image = self.transform(image)
        
        # 使用processor处理图像
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, torch.tensor(label, dtype=torch.long)

class DINOv3ThreeClassModel(nn.Module):
    def __init__(self, dinov3_model_path, num_classes=3):
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
            print(f"Detected ViT model. Feature dim: {self.feature_dim}")
        
        # 三分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 获取DINOv3特征
        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)
        
        if self.model_type == 'convnext':
            last_hidden_state = outputs.last_hidden_state
            features = torch.mean(last_hidden_state, dim=(2, 3))
        else:
            last_hidden_state = outputs.last_hidden_state
            features = last_hidden_state[:, 0, :]
        
        # 通过分类头
        logits = self.classifier(features)
        
        return logits

def calculate_class_weights(labels):
    """计算类别权重用于均衡训练"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # 计算权重：总样本数 / (类别数 * 该类别样本数)
    weights = total_samples / (len(unique_labels) * counts)
    
    class_weights = {}
    for label, weight in zip(unique_labels, weights):
        class_weights[label] = weight
    
    return class_weights

def create_balanced_sampler(dataset):
    """创建均衡采样器"""
    labels = [dataset.dataframe.iloc[i]['label'] for i in range(len(dataset))]
    class_weights = calculate_class_weights(labels)
    
    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def create_output_dir(model_path, balanced=False):
    """创建输出目录"""
    model_name = os.path.basename(model_path)
    mode = "balanced" if balanced else "normal"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_3class/{model_name}_{mode}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    return output_dir

def plot_training_curves(train_history, output_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(train_history['epoch'], train_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(train_history['epoch'], train_history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(train_history['epoch'], train_history['train_acc'], 'b-', label='Train Acc')
    plt.plot(train_history['epoch'], train_history['val_acc'], 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    for i, class_name in enumerate(['Normal', 'Hemo', 'Infarct']):
        plt.plot(train_history['epoch'], train_history[f'val_acc_class_{i}'], 
                label=f'{class_name} Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Class Accuracy')
    plt.title('Per-Class Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    plt.plot(train_history['epoch'], train_history['val_macro_f1'], 'g-', label='Macro F1')
    plt.plot(train_history['epoch'], train_history['val_weighted_f1'], 'm-', label='Weighted F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model(base_path, model_path, balanced=False, random_state=42):
    """训练模型"""
    set_seed(random_state)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据划分
    train_df, test_df = create_train_test_split(base_path, random_state=random_state)
    
    # 创建输出目录
    output_dir = create_output_dir(model_path, balanced=balanced)
    print(f"Output directory: {output_dir}")
    
    # 加载处理器
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # 创建数据集
    train_dataset = ThreeClassDataset(train_df, processor, is_train=True)
    test_dataset = ThreeClassDataset(test_df, processor, is_train=False)
    
    # 创建数据加载器
    if balanced:
        print("使用均衡采样器")
        train_sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
    else:
        print("使用正常采样")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 创建模型
    model = DINOv3ThreeClassModel(model_path, num_classes=3).to(device)
    
    # 损失函数
    if balanced:
        criterion = nn.CrossEntropyLoss()  # 均衡采样时使用普通交叉熵
    else:
        # 正常训练时可以使用加权交叉熵
        train_labels = train_df['label'].values
        class_weights = calculate_class_weights(train_labels)
        weights = torch.tensor([class_weights[i] for i in range(3)], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 优化器
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练历史记录
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_macro_f1': [],
        'val_weighted_f1': [],
        'val_acc_class_0': [],
        'val_acc_class_1': [],
        'val_acc_class_2': []
    }
    
    num_epochs = 100
    best_macro_f1 = 0
    
    # 保存配置
    config_info = f"""Training Configuration:
Model: {os.path.basename(model_path)}
Task: Three-Class Classification (Normal vs Hemo vs Infarct)
Training Mode: {'Balanced Sampling' if balanced else 'Normal Training'}
Device: {device}
Random Seed: {random_state}
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
    
    # 训练循环
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
        from sklearn.metrics import f1_score, accuracy_score
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        val_acc = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # 计算每类准确率
        class_accs = []
        for class_idx in range(3):
            class_mask = all_labels == class_idx
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(all_labels[class_mask], all_predictions[class_mask])
                class_accs.append(class_acc)
            else:
                class_accs.append(0.0)
        
        # 保存训练历史
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(test_loader)
        
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_acc'].append(avg_train_acc)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['val_macro_f1'].append(macro_f1)
        train_history['val_weighted_f1'].append(weighted_f1)
        train_history['val_acc_class_0'].append(class_accs[0])
        train_history['val_acc_class_1'].append(class_accs[1])
        train_history['val_acc_class_2'].append(class_accs[2])
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}')
        print(f'Class Accuracies - Normal: {class_accs[0]:.4f}, Hemo: {class_accs[1]:.4f}, Infarct: {class_accs[2]:.4f}')
        
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            print(f'New best model saved with Macro F1: {best_macro_f1:.4f}')
        
        print('-' * 50)
    
    # 保存训练历史和数据划分
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    
    # 绘制训练曲线
    plot_training_curves(train_history, output_dir)
    
    return output_dir, train_df, test_df

def evaluate_model(output_dir, model_path, test_df):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载处理器和测试数据
    processor = AutoImageProcessor.from_pretrained(model_path)
    test_dataset = ThreeClassDataset(test_df, processor, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载模型
    model = DINOv3ThreeClassModel(model_path, num_classes=3).to(device)
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
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # 计算指标
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    acc = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    class_names = ['Normal', 'Hemo', 'Infarct']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 分类报告
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # 保存详细结果
    results = {
        'overall_metrics': {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall)
        },
        'per_class_metrics': report,
        'confusion_matrix': cm.tolist(),
        'test_predictions': {
            'labels': all_labels.tolist(),
            'predictions': all_predictions.tolist(),
            'probabilities': all_probabilities.tolist()
        }
    }
    
    # 保存结果到JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存概率预测结果
    prob_df = test_df.copy()
    prob_df['predicted_label'] = all_predictions
    prob_df['prob_normal'] = all_probabilities[:, 0]
    prob_df['prob_hemo'] = all_probabilities[:, 1]
    prob_df['prob_infarct'] = all_probabilities[:, 2]
    prob_df.to_csv(os.path.join(output_dir, 'predictions_with_probabilities.csv'), index=False)
    
    # 打印结果
    results_text = f"""Final Evaluation Results:
Model: {os.path.basename(model_path)}
Evaluation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test Samples: {len(test_dataset)}

Overall Metrics:
Accuracy: {acc:.4f} ({acc*100:.2f}%)
Macro F1: {macro_f1:.4f}
Weighted F1: {weighted_f1:.4f}
Macro Precision: {macro_precision:.4f}
Macro Recall: {macro_recall:.4f}

Per-Class Results:
Normal    - Precision: {report['Normal']['precision']:.4f}, Recall: {report['Normal']['recall']:.4f}, F1: {report['Normal']['f1-score']:.4f}
Hemo      - Precision: {report['Hemo']['precision']:.4f}, Recall: {report['Hemo']['recall']:.4f}, F1: {report['Hemo']['f1-score']:.4f}
Infarct   - Precision: {report['Infarct']['precision']:.4f}, Recall: {report['Infarct']['recall']:.4f}, F1: {report['Infarct']['f1-score']:.4f}

Confusion Matrix:
{cm}
"""
    
    print(results_text)
    
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(results_text)
    
    return results

def main():
    """主函数"""
    # 配置
    base_path = "/path/to/Dataset102_3classes"
    model_path = "/path/to/weights/dinov3-vith16plus-pretrain-lvd1689m"
    random_state = 42
    
    print("开始三分类训练...")
    
    # 正常训练模式
    print("\n" + "="*50)
    print("正常训练模式 (按比例训练)")
    print("="*50)
    output_dir_normal, train_df, test_df = train_model(
        base_path, model_path, balanced=False, random_state=random_state
    )
    
    print(f"\n开始正常模式评估...")
    normal_results = evaluate_model(output_dir_normal, model_path, test_df)
    
    # 均衡训练模式
    print("\n" + "="*50)
    print("均衡训练模式 (均衡采样)")
    print("="*50)
    output_dir_balanced, _, _ = train_model(
        base_path, model_path, balanced=True, random_state=random_state
    )
    
    print(f"\n开始均衡模式评估...")
    balanced_results = evaluate_model(output_dir_balanced, model_path, test_df)
    
    # 比较结果
    print("\n" + "="*50)
    print("模式比较")
    print("="*50)
    print(f"正常训练 - 准确率: {normal_results['overall_metrics']['accuracy']:.4f}, Macro F1: {normal_results['overall_metrics']['macro_f1']:.4f}")
    print(f"均衡训练 - 准确率: {balanced_results['overall_metrics']['accuracy']:.4f}, Macro F1: {balanced_results['overall_metrics']['macro_f1']:.4f}")
    
    print(f"\n正常训练结果保存到: {output_dir_normal}")
    print(f"均衡训练结果保存到: {output_dir_balanced}")

if __name__ == "__main__":
    main()