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
import SimpleITK as sitk
from surface_distance import compute_surface_distances, compute_robust_hausdorff

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, processor, is_train=True):
        self.data_dir = data_dir
        self.processor = processor
        self.is_train = is_train
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
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
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 提取原图(R通道)和标签(G通道)
        image_array = np.array(image)
        original_image = Image.fromarray(image_array[:, :, 0]).convert('RGB')  # R通道作为原图
        mask = image_array[:, :, 1]  # G通道作为标签
        
        # 数据增强
        if self.transform and self.is_train:
            # 对原图和mask同时进行相同的变换
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            original_image = self.transform(original_image)
            torch.manual_seed(seed)
            mask = self.transform(Image.fromarray(mask))
            mask = np.array(mask)
        
        # 处理输入图像
        inputs = self.processor(images=original_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # 移除batch维度
        
        # 将mask转换为二值mask (假设非零值为前景)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask)
        
        return pixel_values, mask

class DINOv3MultiLayerSegmentationModel(nn.Module):
    def __init__(self, dinov3_model_path, layer_indices=[2, 5, 8, 11], num_classes=1):
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
        
        print(f"Using layers: {self.layer_indices}")
        print(f"Feature dimension per layer: {self.feature_dim}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Total concatenated feature dimension: {self.feature_dim * self.num_layers}")
        
        # 简单的MLP分割头
        # 输入: concatenated features [B, feature_dim * num_layers, H, W]
        # 输出: segmentation map [B, num_classes, 224, 224]
        
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
            
            # 输出层
            nn.Conv2d(64, num_classes, 1)
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
        # 每个features: [B, feature_dim, 14, 14]
        # 拼接后: [B, feature_dim * num_layers, 14, 14]
        concatenated_features = torch.cat(multi_layer_features, dim=1)
        
        # 通过MLP得到分割结果
        # 输入: [B, feature_dim * num_layers, 14, 14]
        # 输出: [B, num_classes, 14, 14]
        segmentation_output = self.segmentation_mlp(concatenated_features)
        
        # 上采样到目标尺寸 (14x14 -> 224x224)
        # 这里只需要一次上采样，从14x14到224x224，放大16倍
        final_output = F.interpolate(
            segmentation_output, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        return final_output

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def calculate_dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """计算Dice系数"""
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    
    intersection = (pred & target).float().sum((1, 2, 3))
    union = pred.float().sum((1, 2, 3)) + target.float().sum((1, 2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_iou(pred, target, threshold=0.5):
    """计算IoU"""
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    
    intersection = (pred & target).float().sum((1, 2, 3))
    union = (pred | target).float().sum((1, 2, 3))
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def calculate_hd95_fast(pred_mask, true_mask, spacing=(1.0, 1.0)):
    """使用surface-distance库快速计算HD95"""
    try:
        # 转换为numpy数组
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy().astype(bool)
        if isinstance(true_mask, torch.Tensor):
            true_mask = true_mask.cpu().numpy().astype(bool)
        
        # 如果其中一个mask为空，返回特殊值
        if np.sum(pred_mask) == 0 and np.sum(true_mask) == 0:
            return 0.0  # 两个都为空，距离为0
        elif np.sum(pred_mask) == 0 or np.sum(true_mask) == 0:
            return 100.0  # 其中一个为空，返回大值
        
        # 使用surface-distance库计算
        surface_distances = compute_surface_distances(
            true_mask, pred_mask, spacing_mm=spacing)
        hd95 = compute_robust_hausdorff(surface_distances, 95)
        
        return hd95
        
    except Exception as e:
        print(f"HD95计算错误: {e}")
        return 100.0

def create_output_dir(model_path, layer_config):
    """创建输出目录"""
    # 提取模型名称
    model_name = os.path.basename(model_path)
    
    # 创建层配置字符串
    layer_str = "_".join(map(str, layer_config))
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{model_name}_layers_{layer_str}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    return output_dir

def train_model_with_layer_config(layer_indices):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据路径
    train_dir = "/path/to/train"
    test_dir = "/path/to/test"
    
    # 模型路径 - 只测试ViT-B
    model_path = "/path/to/weights/dinov3-vitb16-pretrain-lvd1689m"
    
    # 创建输出目录
    output_dir = create_output_dir(model_path, layer_indices)
    print(f"Output directory: {output_dir}")
    print(f"Using layers: {layer_indices}")
    
    # 加载处理器
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # 创建数据集和数据加载器
    train_dataset = SegmentationDataset(train_dir, processor, is_train=True)
    test_dataset = SegmentationDataset(test_dir, processor, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 创建模型
    model = DINOv3MultiLayerSegmentationModel(model_path, layer_indices=layer_indices, num_classes=1).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练历史记录
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_iou': [],
        'train_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }
    
    # 训练循环
    num_epochs = 100
    best_dice = 0
    
    # 标记是否已经打印过调试信息
    debug_printed = False
    
    # 保存训练配置
    config_info = f"""Training Configuration:
Model: {os.path.basename(model_path)}
Layer Indices: {layer_indices}
Device: {device}
Epochs: {num_epochs}
Batch Size: 8
Learning Rate: 1e-4
Weight Decay: 1e-5
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
        train_iou = 0
        train_dice = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            # 确保masks的维度正确
            if len(masks.shape) == 3:  # [B, H, W]
                masks = masks.unsqueeze(1)  # [B, 1, H, W]
            
            # 调整mask尺寸到224x224以匹配输出
            masks_resized = F.interpolate(masks, size=(224, 224), mode='nearest')
            
            optimizer.zero_grad()
            
            # 只在第一个batch打印调试信息
            if not debug_printed:
                print(f"\nDebugging shapes:")
                print(f"Images shape: {images.shape}")
                print(f"Original masks shape: {masks.shape}")
                print(f"Resized masks shape: {masks_resized.shape}")
                
                outputs = model(images)
                print(f"Model output shape: {outputs.shape}")
                
                # 打印中间特征的形状
                print("Debug completed. Continuing training...")
                debug_printed = True
            else:
                outputs = model(images)
            
            # 计算损失
            bce_loss = criterion(outputs, masks_resized)
            dice_loss_val = dice_loss(outputs, masks_resized)
            loss = bce_loss + dice_loss_val
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks_resized).item()
            train_dice += calculate_dice_score(outputs, masks_resized).item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{calculate_iou(outputs, masks_resized).item():.4f}',
                'Dice': f'{calculate_dice_score(outputs, masks_resized).item():.4f}'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                
                # 确保masks的维度正确
                if len(masks.shape) == 3:  # [B, H, W]
                    masks = masks.unsqueeze(1)  # [B, 1, H, W]
                
                # 调整mask尺寸
                masks_resized = F.interpolate(masks, size=(224, 224), mode='nearest')
                
                outputs = model(images)
                
                bce_loss = criterion(outputs, masks_resized)
                dice_loss_val = dice_loss(outputs, masks_resized)
                loss = bce_loss + dice_loss_val
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks_resized).item()
                val_dice += calculate_dice_score(outputs, masks_resized).item()
        
        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        avg_val_iou = val_iou / len(test_loader)
        avg_val_dice = val_dice / len(test_loader)
        
        # 保存训练历史
        train_history['epoch'].append(epoch + 1)
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_iou'].append(avg_train_iou)
        train_history['train_dice'].append(avg_train_dice)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_iou'].append(avg_val_iou)
        train_history['val_dice'].append(avg_val_dice)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train Dice: {avg_train_dice:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}')
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型（基于Dice分数）
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            print(f'New best model saved with Dice: {best_dice:.4f}')
        
        # 每200个epoch保存一次checkpoint
        if (epoch + 1) % 200 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'train_history': train_history,
                'layer_indices': layer_indices
            }, os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth'))
        
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
    plt.plot(train_history['epoch'], train_history['train_iou'], 'b-', label='Train IoU')
    plt.plot(train_history['epoch'], train_history['val_iou'], 'r-', label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_history['epoch'], train_history['train_dice'], 'b-', label='Train Dice')
    plt.plot(train_history['epoch'], train_history['val_dice'], 'r-', label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Training and Validation Dice')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir, best_dice

def evaluate_model_multilayer(output_dir, model_path, layer_indices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据路径
    test_dir = "/path/to/test"
    
    # 加载处理器和测试数据
    processor = AutoImageProcessor.from_pretrained(model_path)
    test_dataset = SegmentationDataset(test_dir, processor, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    model = DINOv3MultiLayerSegmentationModel(model_path, layer_indices=layer_indices, num_classes=1).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoints', 'best_model.pth')))
    model.eval()
    
    # 评估指标
    total_iou = 0
    total_dice = 0
    total_hd95 = 0
    valid_hd95_count = 0
    
    print("Starting detailed evaluation...")
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images, masks = images.to(device), masks.to(device)
            
            # 确保masks的维度正确
            if len(masks.shape) == 3:  # [B, H, W]
                masks = masks.unsqueeze(1)  # [B, 1, H, W]
            
            # 调整mask尺寸
            masks_224 = F.interpolate(masks, size=(224, 224), mode='nearest')
            
            outputs = model(images)
            pred_mask = torch.sigmoid(outputs) > 0.5
            
            # 计算IoU和Dice
            iou = calculate_iou(outputs, masks_224)
            dice = calculate_dice_score(outputs, masks_224)
            total_iou += iou.item()
            total_dice += dice.item()
            
            # 使用快速HD95计算
            pred_np = pred_mask[0, 0].cpu().numpy()
            true_np = (masks_224[0, 0] > 0.5).cpu().numpy()
            
            hd95 = calculate_hd95_fast(pred_np, true_np)
            if hd95 < 100.0:
                total_hd95 += hd95
                valid_hd95_count += 1
            
            # 可视化前5个结果
            if idx < 5:
                plt.figure(figsize=(25, 5))
                
                # 原图
                plt.subplot(1, 5, 1)
                img_np = images[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                plt.imshow(img_np)
                plt.title('Original Image (224x224)')
                plt.axis('off')
                
                # 原始尺寸的真实标签
                plt.subplot(1, 5, 2)
                plt.imshow(masks[0, 0].cpu().numpy(), cmap='gray')
                plt.title('Ground Truth (Original Size)')
                plt.axis('off')
                
                # 调整尺寸后的真实标签
                plt.subplot(1, 5, 3)
                plt.imshow(masks_224[0, 0].cpu().numpy(), cmap='gray')
                plt.title('Ground Truth (224x224)')
                plt.axis('off')
                
                # 预测结果
                plt.subplot(1, 5, 4)
                plt.imshow(pred_mask[0, 0].cpu().numpy(), cmap='gray')
                plt.title('Prediction (224x224)')
                plt.axis('off')
                
                # 重叠显示
                plt.subplot(1, 5, 5)
                pred_np_vis = pred_mask[0, 0].cpu().numpy()
                true_np_vis = (masks_224[0, 0] > 0.5).cpu().numpy()
                
                # 创建RGB图像显示重叠
                overlap = np.zeros((pred_np_vis.shape[0], pred_np_vis.shape[1], 3))
                overlap[:, :, 0] = pred_np_vis  # 预测为红色
                overlap[:, :, 1] = true_np_vis  # 真实为绿色
                
                plt.imshow(overlap)
                plt.title('Overlap (Red: Pred, Green: GT)')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'visualizations', f'result_{idx:03d}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
    
    # 计算平均指标
    avg_iou = total_iou / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    avg_hd95 = total_hd95 / valid_hd95_count if valid_hd95_count > 0 else float('inf')
    
    results_text = f"""Final Evaluation Results:
Model: {os.path.basename(model_path)}
Layer Indices: {layer_indices}
Evaluation Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test Samples: {len(test_loader)}

Metrics:
Average IoU: {avg_iou:.4f}
Average Dice: {avg_dice:.4f}
Average HD95: {avg_hd95:.4f} (computed on {valid_hd95_count}/{len(test_loader)} valid samples)

Notes:
- IoU (Intersection over Union): Higher is better (0-1)
- Dice Score: Higher is better (0-1)
- HD95 (Hausdorff Distance 95th percentile): Lower is better (pixels)
"""
    
    print(results_text)
    
    # 保存结果到文件
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(results_text)
    
    return avg_iou, avg_dice, avg_hd95

if __name__ == "__main__":
    # 定义不同的层配置进行实验
    layer_configs = [
        [11],                    # 只使用最后一层
        [2, 5, 8, 11],          # 4层配置
        [1, 3, 5, 7, 9, 11],    # 6层配置
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    
    results_summary = []
    
    print("Starting multi-layer experiments...")
    
    for i, layer_indices in enumerate(layer_configs):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(layer_configs)}: Layers {layer_indices}")
        print(f"{'='*60}")
        
        try:
            # 训练模型
            output_dir, best_dice = train_model_with_layer_config(layer_indices)
            
            # 评估模型
            model_path = "/path/to/weights/dinov3-vitb16-pretrain-lvd1689m"
            avg_iou, avg_dice, avg_hd95 = evaluate_model_multilayer(output_dir, model_path, layer_indices)
            
            # 记录结果
            results_summary.append({
                'layers': layer_indices,
                'best_dice': best_dice,
                'final_iou': avg_iou,
                'final_dice': avg_dice,
                'final_hd95': avg_hd95,
                'output_dir': output_dir
            })
            
            print(f"Experiment {i+1} completed successfully!")
            
        except Exception as e:
            print(f"Experiment {i+1} failed with error: {e}")
            continue
    
    # 保存实验总结
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    summary_text = "Multi-Layer DINOv3 Segmentation Experiments Summary\n"
    summary_text += "="*60 + "\n\n"
    
    for i, result in enumerate(results_summary):
        summary_text += f"Experiment {i+1}:\n"
        summary_text += f"  Layers: {result['layers']}\n"
        summary_text += f"  Best Training Dice: {result['best_dice']:.4f}\n"
        summary_text += f"  Final IoU: {result['final_iou']:.4f}\n"
        summary_text += f"  Final Dice: {result['final_dice']:.4f}\n"
        summary_text += f"  Final HD95: {result['final_hd95']:.4f}\n"
        summary_text += f"  Output Dir: {result['output_dir']}\n"
        summary_text += "-" * 40 + "\n"
        
        print(f"Layers {result['layers']}: Dice={result['final_dice']:.4f}, IoU={result['final_iou']:.4f}, HD95={result['final_hd95']:.4f}")
    
    # 保存总结到文件
    with open('multi_layer_experiments_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"\nAll experiments completed! Summary saved to 'multi_layer_experiments_summary.txt'")