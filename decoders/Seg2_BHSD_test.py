# BHSD Multi-class Multi-layer Segmentation: EDH, IPH, IVA, SAH, SDH (Fixed Version)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import gc

class BHSDMultiLayerSegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, processor, file_list=None, is_train=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.processor = processor
        self.is_train = is_train
        
        # 如果提供了文件列表，使用它；否则使用所有文件
        if file_list is not None:
            self.image_files = file_list
        else:
            self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        
        # 血肿类型标签映射
        self.label_mapping = {
            'EDH': 1,
            'IPH': 2, 
            'IVA': 3,
            'SAH': 4,
            'SDH': 5
        }
        
        # 数据增强
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        
        # 加载原图
        image = Image.open(img_path).convert('RGB')
        
        # 创建多类标签mask (0=背景, 1=EDH, 2=IPH, 3=IVA, 4=SAH, 5=SDH)
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)  # [H, W]
        
        # 为每种血肿类型加载对应的标签
        for hemo_type, label_value in self.label_mapping.items():
            label_path = os.path.join(self.labels_dir, hemo_type, img_filename)
            if os.path.exists(label_path):
                hemo_mask = np.array(Image.open(label_path).convert('L'))
                # 将该类型的区域标记为对应的标签值
                mask[hemo_mask > 127] = label_value
        
        # 数据增强
        if self.transform and self.is_train:
            # 对原图和mask同时进行相同的变换
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask_pil = Image.fromarray(mask)
            mask_pil = self.transform(mask_pil)
            mask = np.array(mask_pil)
        
        # 处理输入图像
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # 移除batch维度
        
        # 转换mask为tensor
        mask = torch.from_numpy(mask).long()
        
        return pixel_values, mask

class DINOv3MultiLayerMultiClassSegmentationModel(nn.Module):
    def __init__(self, dinov3_model_path, layer_indices=[2, 5, 8, 11], num_classes=6):  # 6类: 背景 + 5种血肿
        super().__init__()
        
        # 加载DINOv3模型
        self.dinov3 = AutoModel.from_pretrained(dinov3_model_path)
        
        # 冻结DINOv3参数
        for param in self.dinov3.parameters():
            param.requires_grad = False
        
        # 验证层索引
        num_transformer_layers = self.dinov3.config.num_hidden_layers
        print(f"DINOv3 has {num_transformer_layers} transformer layers")
        
        # 调整层索引（确保在有效范围内）
        self.layer_indices = [idx for idx in layer_indices if 0 <= idx < num_transformer_layers]
        if len(self.layer_indices) != len(layer_indices):
            print(f"Warning: Some layer indices were invalid. Using: {self.layer_indices}")
        
        self.feature_dim = self.dinov3.config.hidden_size
        self.patch_size = 16
        self.num_class_tokens = 1
        self.num_register_tokens = 4
        self.num_layers = len(self.layer_indices)
        self.num_classes = num_classes
        
        print(f"Using layers: {self.layer_indices}")
        print(f"Feature dimension per layer: {self.feature_dim}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Total concatenated feature dimension: {self.feature_dim * self.num_layers}")
        print(f"Number of output classes: {self.num_classes}")
        
        # 多类分割MLP
        concat_dim = self.feature_dim * self.num_layers
        
        self.segmentation_mlp = nn.Sequential(
            # 第一层：降维
            nn.Conv2d(concat_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第二层
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第三层
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第四层
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 输出层 - 输出多类概率
            nn.Conv2d(64, self.num_classes, 1)
        )
        
        print(f"MLP input channels: {concat_dim}")
    
    def extract_patch_features(self, hidden_state, input_height, input_width):
        """从hidden state中提取patch特征并重构为空间特征图"""
        # hidden_state: [B, N, D]
        
        # 分离patch tokens
        patch_tokens_start = self.num_class_tokens + self.num_register_tokens
        patch_tokens = hidden_state[:, patch_tokens_start:, :]  # [B, num_patches, D]
        
        # 计算patch的空间维度
        feature_h = input_height // self.patch_size  # 224 // 16 = 14
        feature_w = input_width // self.patch_size   # 224 // 16 = 14
        expected_patches = feature_h * feature_w     # 14 * 14 = 196
        
        # 验证patch数量
        actual_patches = patch_tokens.shape[1]
        if actual_patches != expected_patches:
            print(f"Warning: Expected {expected_patches} patches, got {actual_patches}")
            feature_h = feature_w = int(math.sqrt(actual_patches))
        
        # 重塑为空间特征图: [B, num_patches, D] -> [B, D, H, W]
        B, N, D = patch_tokens.shape
        features = patch_tokens.view(B, feature_h, feature_w, D).permute(0, 3, 1, 2)
        
        return features
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 获取所有隐藏状态
        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states  # List of [B, N, D]
        
        # 提取多层特征
        multi_layer_features = []
        
        for layer_idx in self.layer_indices:
            hidden_state = all_hidden_states[layer_idx]  # [B, N, D]
            features = self.extract_patch_features(hidden_state, H, W)  # [B, D, 14, 14]
            multi_layer_features.append(features)
        
        # 沿通道维度拼接所有层的特征
        concatenated_features = torch.cat(multi_layer_features, dim=1)
        
        # 通过MLP得到多类分割结果
        segmentation_output = self.segmentation_mlp(concatenated_features)
        
        # 上采样到目标尺寸 (14x14 -> 224x224)
        final_output = F.interpolate(
            segmentation_output, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        return final_output

def create_train_test_split(images_dir, test_size=0.2, random_state=42):
    """创建训练测试数据划分"""
    # 获取所有图像文件
    all_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    # 划分训练测试集
    train_files, test_files = train_test_split(
        all_files, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Dataset split:")
    print(f"Total files: {len(all_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    return train_files, test_files

def multiclass_dice_loss(pred, target, num_classes=6, smooth=1e-6):
    """多类Dice损失"""
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    dice_loss = 0
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target_one_hot[:, c]
        
        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss += (1 - dice.mean())
    
    return dice_loss / num_classes

def calculate_multiclass_dice(pred, target, num_classes=6, smooth=1e-6):
    """计算多类Dice分数"""
    pred = F.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred, dim=1)
    
    dice_scores = []
    for c in range(1, num_classes):  # 跳过背景类
        pred_c = (pred_labels == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())
        else:
            dice_scores.append(1.0)
    
    return dice_scores

def calculate_multiclass_iou(pred, target, num_classes=6):
    """计算多类IoU"""
    pred = F.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred, dim=1)
    
    iou_scores = []
    for c in range(1, num_classes):  # 跳过背景类
        pred_c = (pred_labels == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).float().sum()
        union = (pred_c | target_c).float().sum()
        
        if union > 0:
            iou = intersection / union
            iou_scores.append(iou.item())
        else:
            iou_scores.append(1.0)
    
    return iou_scores

def create_output_dir(model_path, layer_config):
    """创建输出目录"""
    model_name = os.path.basename(model_path)
    layer_str = "_".join(map(str, layer_config))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_bhsd_multilayer_multiclass_seg/{model_name}_layers_{layer_str}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    return output_dir

def plot_training_curves(train_history, output_dir, layer_indices):
    """绘制训练曲线"""
    class_names = ['EDH', 'IPH', 'IVA', 'SAH', 'SDH']
    
    plt.figure(figsize=(20, 15))
    
    # 总体损失和准确率
    plt.subplot(3, 4, 1)
    plt.plot(train_history['epoch'], train_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(train_history['epoch'], train_history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Multi-layer Loss (Layers: {layer_indices})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 4, 2)
    plt.plot(train_history['epoch'], train_history['train_acc'], 'b-', label='Train Acc')
    plt.plot(train_history['epoch'], train_history['val_acc'], 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Multi-layer Pixel Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 平均Dice和IoU
    plt.subplot(3, 4, 3)
    plt.plot(train_history['epoch'], train_history['train_mean_dice'], 'b-', label='Train Dice')
    plt.plot(train_history['epoch'], train_history['val_mean_dice'], 'r-', label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Dice')
    plt.title('Multi-layer Mean Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 4, 4)
    plt.plot(train_history['epoch'], train_history['train_mean_iou'], 'b-', label='Train IoU')
    plt.plot(train_history['epoch'], train_history['val_mean_iou'], 'r-', label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Multi-layer Mean IoU Score')
    plt.legend()
    plt.grid(True)
    
    # 每类Dice分数
    for i, class_name in enumerate(class_names):
        plt.subplot(3, 4, 5 + i)
        plt.plot(train_history['epoch'], train_history[f'val_dice_{class_name}'], 
                label=f'{class_name} Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title(f'Multi-layer {class_name} Dice')
        plt.legend()
        plt.grid(True)
    
    # 学习率
    if 'learning_rate' in train_history:
        plt.subplot(3, 4, 10)
        plt.plot(train_history['epoch'], train_history['learning_rate'], 'purple')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_multilayer.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model_with_layer_config(layer_indices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据路径
    images_dir = "/path/to/segmentation/images"
    labels_dir = "/path/to/segmentation/labels"
    
    # 模型路径
    model_path = "/path/to/weights/dinov3-vitb16-pretrain-lvd1689m"
    
    # 创建输出目录
    output_dir = create_output_dir(model_path, layer_indices)
    print(f"Output directory: {output_dir}")
    print(f"Using layers: {layer_indices}")
    
    # 显式创建训练测试划分
    train_files, test_files = create_train_test_split(images_dir, test_size=0.2, random_state=42)
    
    # 保存文件列表
    with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
        for filename in train_files:
            f.write(f"{filename}\n")
    
    with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
        for filename in test_files:
            f.write(f"{filename}\n")
    
    # 加载处理器
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # 创建数据集 - 使用文件列表
    train_dataset = BHSDMultiLayerSegmentationDataset(images_dir, labels_dir, processor, 
                                                    file_list=train_files, is_train=True)
    test_dataset = BHSDMultiLayerSegmentationDataset(images_dir, labels_dir, processor, 
                                                   file_list=test_files, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 创建模型
    model = DINOv3MultiLayerMultiClassSegmentationModel(
        model_path, layer_indices=layer_indices, num_classes=6
    ).to(device)
    
    # 计算类别权重
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)  # 背景权重较小
    
    # 损失函数和优化器
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=7)
    
    # 训练历史记录
    class_names = ['EDH', 'IPH', 'IVA', 'SAH', 'SDH']
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_mean_dice': [],
        'train_mean_iou': [],
        'val_loss': [],
        'val_acc': [],
        'val_mean_dice': [],
        'val_mean_iou': [],
        'learning_rate': []
    }
    
    # 为每个类别添加指标
    for class_name in class_names:
        train_history[f'val_dice_{class_name}'] = []
        train_history[f'val_iou_{class_name}'] = []
    
    num_epochs = 100
    best_dice = 0
    patience_counter = 0
    early_stopping_patience = 50
    debug_printed = False
    
    # 保存配置
    config_info = f"""BHSD Multi-class Multi-layer Segmentation Training Configuration:
Model: {os.path.basename(model_path)}
Layer Indices: {layer_indices}
Task: Multi-class Multi-layer Segmentation (Background + EDH + IPH + IVA + SAH + SDH)
Device: {device}
Epochs: {num_epochs}
Batch Size: 8
Learning Rate: 1e-4
Weight Decay: 1e-5
Early Stopping Patience: {early_stopping_patience}
Train Samples: {len(train_dataset)}
Test Samples: {len(test_dataset)}
Class Weights: {class_weights.tolist()}
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
        train_dice_sum = [0] * 5  # 5个血肿类别
        train_iou_sum = [0] * 5
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            # 调整mask尺寸到224x224
            masks_resized = F.interpolate(masks.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()
            
            optimizer.zero_grad()
            
            if not debug_printed:
                print(f"\nDebugging multi-layer shapes:")
                print(f"Images shape: {images.shape}")
                print(f"Original masks shape: {masks.shape}")
                print(f"Resized masks shape: {masks_resized.shape}")
                print(f"Unique labels in mask: {torch.unique(masks_resized)}")
                
                outputs = model(images)
                print(f"Model output shape: {outputs.shape}")
                print("Multi-layer debug completed. Continuing training...")
                debug_printed = True
            else:
                outputs = model(images)
            
            # 计算损失
            ce_loss = ce_criterion(outputs, masks_resized)
            dice_loss_val = multiclass_dice_loss(outputs, masks_resized, num_classes=6)
            loss = ce_loss + dice_loss_val
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算像素准确率
            pred_labels = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct = (pred_labels == masks_resized).float().sum()
            total = masks_resized.numel()
            train_correct += correct.item()
            train_total += total
            
            # 计算Dice和IoU
            dice_scores = calculate_multiclass_dice(outputs, masks_resized)
            iou_scores = calculate_multiclass_iou(outputs, masks_resized)
            
            for i in range(5):
                train_dice_sum[i] += dice_scores[i]
                train_iou_sum[i] += iou_scores[i]
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%',
                'Mean Dice': f'{np.mean(dice_scores):.4f}'
            })
            
            # 清理内存
            del outputs, loss, ce_loss, dice_loss_val
            torch.cuda.empty_cache()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_dice_sum = [0] * 5
        val_iou_sum = [0] * 5
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                masks_resized = F.interpolate(masks.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()
                
                outputs = model(images)
                
                ce_loss = ce_criterion(outputs, masks_resized)
                dice_loss_val = multiclass_dice_loss(outputs, masks_resized, num_classes=6)
                loss = ce_loss + dice_loss_val
                
                val_loss += loss.item()
                
                # 像素准确率
                pred_labels = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                correct = (pred_labels == masks_resized).float().sum()
                total = masks_resized.numel()
                val_correct += correct.item()
                val_total += total
                
                # Dice和IoU
                dice_scores = calculate_multiclass_dice(outputs, masks_resized)
                iou_scores = calculate_multiclass_iou(outputs, masks_resized)
                
                for i in range(5):
                    val_dice_sum[i] += dice_scores[i]
                    val_iou_sum[i] += iou_scores[i]
                
                # 清理内存
                del outputs, loss
                torch.cuda.empty_cache()
        
        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_train_dice = [dice_sum / len(train_loader) for dice_sum in train_dice_sum]
        avg_train_iou = [iou_sum / len(train_loader) for iou_sum in train_iou_sum]
        
        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_correct / val_total
        avg_val_dice = [dice_sum / len(test_loader) for dice_sum in val_dice_sum]
        avg_val_iou = [iou_sum / len(test_loader) for iou_sum in val_iou_sum]
        
        mean_val_dice = np.mean(avg_val_dice)
        mean_val_iou = np.mean(avg_val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存训练历史
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_acc'].append(avg_train_acc)
        train_history['train_mean_dice'].append(np.mean(avg_train_dice))
        train_history['train_mean_iou'].append(np.mean(avg_train_iou))
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_acc'].append(avg_val_acc)
        train_history['val_mean_dice'].append(mean_val_dice)
        train_history['val_mean_iou'].append(mean_val_iou)
        train_history['learning_rate'].append(current_lr)
        
        for i, class_name in enumerate(class_names):
            train_history[f'val_dice_{class_name}'].append(avg_val_dice[i])
            train_history[f'val_iou_{class_name}'].append(avg_val_iou[i])
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        print(f'Mean Val Dice: {mean_val_dice:.4f}, Mean Val IoU: {mean_val_iou:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        print("Per-class multi-layer Dice scores:")
        for i, class_name in enumerate(class_names):
            print(f'  {class_name}: {avg_val_dice[i]:.4f}')
        
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型和早停
        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            print(f'New best multi-layer model saved with Mean Dice: {best_dice:.4f}')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        print('-' * 80)
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(train_history, output_dir, layer_indices)
    
    return output_dir, best_dice, test_files

def evaluate_model_multilayer_multiclass(output_dir, model_path, layer_indices, test_files):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据路径
    images_dir = "/path/to/segmentation/images"
    labels_dir = "/path/to/segmentation/labels"
    
    # 加载处理器和测试数据 - 只使用测试文件
    processor = AutoImageProcessor.from_pretrained(model_path)
    test_dataset = BHSDMultiLayerSegmentationDataset(images_dir, labels_dir, processor, 
                                                   file_list=test_files, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)  # 减少batch size和workers
    
    # 加载模型
    model = DINOv3MultiLayerMultiClassSegmentationModel(
        model_path, layer_indices=layer_indices, num_classes=6
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoints', 'best_model.pth')))
    model.eval()
    
    class_names = ['Background', 'EDH', 'IPH', 'IVA', 'SAH', 'SDH']
    hemo_names = ['EDH', 'IPH', 'IVA', 'SAH', 'SDH']
    
    # 评估指标
    total_dice = [0] * 5  # 5个血肿类别
    total_iou = [0] * 5
    class_counts = [0] * 5
    
    all_predictions = []
    all_targets = []
    
    print(f"Starting multi-class multi-layer segmentation evaluation on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images, masks = images.to(device), masks.to(device)
            masks_224 = F.interpolate(masks.unsqueeze(1).float(), size=(224, 224), mode='nearest').squeeze(1).long()
            
            outputs = model(images)
            pred_probs = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=1)
            
            # 收集预测和真实标签用于混淆矩阵 (采样以避免内存问题)
            if idx % 10 == 0:  # 每10个样本采样一次
                all_predictions.extend(pred_labels.cpu().numpy().flatten()[::100])  # 进一步采样像素
                all_targets.extend(masks_224.cpu().numpy().flatten()[::100])
            
            # 计算每类的Dice和IoU
            dice_scores = calculate_multiclass_dice(outputs, masks_224)
            iou_scores = calculate_multiclass_iou(outputs, masks_224)
            
            # 检查每个类别是否存在
            for c in range(1, 6):  # 跳过背景类
                if torch.sum(masks_224 == c) > 0:
                    total_dice[c-1] += dice_scores[c-1]
                    total_iou[c-1] += iou_scores[c-1]
                    class_counts[c-1] += 1
            
            # 可视化前5个结果
            if idx < 5:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # 原图
                img_np = images[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                axes[0].imshow(img_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # 真实标签
                true_mask = masks_224[0].cpu().numpy()
                axes[1].imshow(true_mask, cmap='tab10', vmin=0, vmax=5)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # 预测结果
                pred_mask = pred_labels[0].cpu().numpy()
                axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=5)
                axes[2].set_title('Multi-layer Prediction')
                axes[2].axis('off')
                
                # 错误图
                error_mask = (true_mask != pred_mask).astype(float)
                axes[3].imshow(error_mask, cmap='Reds')
                axes[3].set_title('Prediction Errors')
                axes[3].axis('off')
                
                fig.suptitle(f'Multi-layer Sample {idx+1} (Layers: {layer_indices}): {class_names}', fontsize=14)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'visualizations', f'multilayer_multiclass_result_{idx:03d}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
            
            # 清理内存
            del outputs, pred_probs, pred_labels
            torch.cuda.empty_cache()
    
    # 计算平均指标
    avg_dice = [total_dice[i] / max(class_counts[i], 1) for i in range(5)]
    avg_iou = [total_iou[i] / max(class_counts[i], 1) for i in range(5)]
    
    # 计算混淆矩阵 (如果有采样的数据)
    if len(all_predictions) > 0 and len(all_targets) > 0:
        cm = confusion_matrix(all_targets, all_predictions, labels=list(range(6)))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Multi-class Multi-layer Segmentation Confusion Matrix\n(Layers: {layer_indices})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multilayer_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        cm = np.zeros((6, 6))
    
    # 生成结果报告
    results_text = f"""BHSD Multi-class Multi-layer Segmentation Evaluation Results:
Model: {os.path.basename(model_path)}
Layer Indices: {layer_indices}
Evaluation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test Samples: {len(test_dataset)}

Per-Class Metrics:
"""
    
    for i, class_name in enumerate(hemo_names):
        samples_with_class = class_counts[i]
        dice_score = avg_dice[i]
        iou_score = avg_iou[i]
        results_text += f"{class_name:4s} - Dice: {dice_score:.4f}, IoU: {iou_score:.4f}, Samples: {samples_with_class}\n"
    
    mean_dice = np.mean(avg_dice)
    mean_iou = np.mean(avg_iou)
    
    results_text += f"""
Overall Metrics:
Mean Dice: {mean_dice:.4f}
Mean IoU: {mean_iou:.4f}

Notes:
- Multi-layer feature fusion from layers: {layer_indices}
- Dice Score: Higher is better (0-1)
- IoU (Intersection over Union): Higher is better (0-1)
- Metrics calculated only on samples containing each class
"""
    
    print(results_text)
    
    # 保存结果
    with open(os.path.join(output_dir, 'multilayer_evaluation_results.txt'), 'w') as f:
        f.write(results_text)
    
    # 保存详细结果到JSON
    results_dict = {
        'architecture': 'DINOv3_MultiLayer_MultiClass',
        'layer_indices': layer_indices,
        'per_class_dice': {hemo_names[i]: avg_dice[i] for i in range(5)},
        'per_class_iou': {hemo_names[i]: avg_iou[i] for i in range(5)},
        'per_class_samples': {hemo_names[i]: class_counts[i] for i in range(5)},
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(output_dir, 'multilayer_evaluation_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return mean_dice, mean_iou

if __name__ == "__main__":
    # 定义不同的层配置进行实验
    layer_configs = [
        [11],                    # 只使用最后一层
        [2, 5, 8, 11],          # 4层配置
        [1, 3, 5, 7, 9, 11],    # 6层配置
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 所有层
    ]
    
    results_summary = []
    
    print("Starting BHSD multi-class multi-layer segmentation experiments...")
    
    for i, layer_indices in enumerate(layer_configs):
        print(f"\n{'='*60}")
        print(f"Multi-layer Experiment {i+1}/{len(layer_configs)}: Layers {layer_indices}")
        print(f"{'='*60}")
        
        try:
            # 训练模型
            output_dir, best_dice, test_files = train_model_with_layer_config(layer_indices)
            
            # 评估模型
            model_path = "/path/to/weights/dinov3-vitb16-pretrain-lvd1689m"
            avg_dice, avg_iou = evaluate_model_multilayer_multiclass(output_dir, model_path, layer_indices, test_files)
            
            # 记录结果
            results_summary.append({
                'layers': layer_indices,
                'best_dice': best_dice,
                'final_dice': avg_dice,
                'final_iou': avg_iou,
                'output_dir': output_dir
            })
            
            print(f"Multi-layer Experiment {i+1} completed successfully!")
            
        except Exception as e:
            print(f"Multi-layer Experiment {i+1} failed with error: {e}")
            continue
    
    # 保存实验总结
    print(f"\n{'='*60}")
    print("BHSD MULTI-CLASS MULTI-LAYER EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    summary_text = "BHSD Multi-Class Multi-Layer DINOv3 Segmentation Experiments Summary\n"
    summary_text += "="*60 + "\n\n"
    
    for i, result in enumerate(results_summary):
        summary_text += f"Multi-layer Experiment {i+1}:\n"
        summary_text += f"  Layers: {result['layers']}\n"
        summary_text += f"  Best Training Dice: {result['best_dice']:.4f}\n"
        summary_text += f"  Final Dice: {result['final_dice']:.4f}\n"
        summary_text += f"  Final IoU: {result['final_iou']:.4f}\n"
        summary_text += f"  Output Dir: {result['output_dir']}\n"
        summary_text += "-" * 40 + "\n"
        
        print(f"Multi-layer {result['layers']}: Dice={result['final_dice']:.4f}, IoU={result['final_iou']:.4f}")
    
    # 保存总结到文件
    with open('bhsd_multilayer_multiclass_experiments_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"\nAll BHSD multi-class multi-layer experiments completed!")
    print(f"Summary saved to 'bhsd_multilayer_multiclass_experiments_summary.txt'")