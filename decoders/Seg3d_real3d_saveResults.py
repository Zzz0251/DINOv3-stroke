# 2d extract, real 3d segmentor - Simplified with Results Saving (Memory Optimized)
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
import nibabel as nib
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
import json
import gc

class SegmentationDataset3D(Dataset):
    def __init__(self, data_dir, label_dir, processor, is_train=True, num_slices=16):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.processor = processor
        self.is_train = is_train
        self.num_slices = num_slices
        
        # 获取所有的nii.gz文件并建立映射
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        self.valid_pairs = []
        
        for data_file in self.data_files:
            if '_0000.nii.gz' in data_file:
                label_file = data_file.replace('_0000.nii.gz', '.nii.gz')
            else:
                label_file = data_file
            
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                self.valid_pairs.append((data_file, label_file))
            else:
                print(f"Warning: No matching label found for {data_file}")
        
        print(f"Found {len(self.valid_pairs)} valid data-label pairs")
        
        # 数据增强
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
            ])
        else:
            self.transform = None
    
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
    
    def sample_slices(self, volume, mask):
        """从3D volume中采样固定数量的切片"""
        depth = volume.shape[0]
        
        if depth <= self.num_slices:
            indices = np.linspace(0, depth-1, self.num_slices, dtype=int)
        else:
            indices = np.linspace(0, depth-1, self.num_slices, dtype=int)
        
        sampled_volume = volume[indices]
        sampled_mask = mask[indices]
        
        return sampled_volume, sampled_mask
    
    def get_original_info(self, idx):
        """获取原始数据信息（不加载完整数据）"""
        data_file, label_file = self.valid_pairs[idx]
        data_path = os.path.join(self.data_dir, data_file)
        label_path = os.path.join(self.label_dir, label_file)
        
        return data_path, label_path, data_file, label_file
    
    def load_original_data(self, idx):
        """单独加载原始数据"""
        data_path, label_path, data_file, label_file = self.get_original_info(idx)
        
        try:
            data_nii = nib.load(data_path)
            label_nii = nib.load(label_path)
            
            volume = data_nii.get_fdata()
            mask = label_nii.get_fdata()
            
            return volume, mask, data_nii, label_nii, data_file, label_file
        except Exception as e:
            print(f"Error loading original data {data_file}: {e}")
            return None, None, None, None, data_file, label_file
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        data_file, label_file = self.valid_pairs[idx]
        
        data_path = os.path.join(self.data_dir, data_file)
        label_path = os.path.join(self.label_dir, label_file)
        
        try:
            data_nii = nib.load(data_path)
            label_nii = nib.load(label_path)
            
            volume = data_nii.get_fdata()
            mask = label_nii.get_fdata()
            
            # 转换维度顺序为 [D, H, W]
            volume = np.transpose(volume, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
        
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            volume = np.zeros((16, 224, 224))
            mask = np.zeros((16, 224, 224))
        
        # 标准化和采样
        volume = self.normalize_to_0_255(volume)
        volume, mask = self.sample_slices(volume, mask)
        volume = self.resize_volume(volume, (224, 224))
        mask = self.resize_volume(mask, (224, 224))
        mask = (mask > 0).astype(np.float32)
        
        # 为每个切片准备DINOv3输入
        slice_features = []
        for i in range(self.num_slices):
            slice_img = Image.fromarray(volume[i]).convert('RGB')
            
            # 数据增强
            if self.transform and self.is_train:
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                slice_img = self.transform(slice_img)
                torch.manual_seed(seed)
                mask_slice = self.transform(Image.fromarray(mask[i]))
                mask[i] = np.array(mask_slice)
            
            inputs = self.processor(images=slice_img, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            slice_features.append(pixel_values)
        
        volume_features = torch.stack(slice_features)  # [num_slices, 3, 224, 224]
        mask_tensor = torch.from_numpy(mask)  # [num_slices, 224, 224]
        
        return volume_features, mask_tensor, idx

class DINOv3Segmentation3D_Simple(nn.Module):
    def __init__(self, dinov3_model_path, num_classes=1, num_slices=16):
        super().__init__()
        
        self.num_slices = num_slices
        
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
        
        # 2D特征降维模块
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 简单的3D卷积块
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final_conv = nn.Conv3d(32, num_classes, 1)
    
    def extract_2d_features(self, x):
        """从2D切片批量提取DINOv3特征"""
        B, C, H, W = x.shape
        
        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)
        
        if self.model_type == 'convnext':
            features = outputs.last_hidden_state
        else:
            last_hidden_state = outputs.last_hidden_state
            patch_tokens_start = self.num_class_tokens + self.num_register_tokens
            patch_tokens = last_hidden_state[:, patch_tokens_start:, :]
            
            feature_h = H // self.patch_size
            feature_w = W // self.patch_size
            expected_patches = feature_h * feature_w
            
            actual_patches = patch_tokens.shape[1]
            if actual_patches != expected_patches:
                feature_h = feature_w = int(math.sqrt(actual_patches))
            
            B, N, D = patch_tokens.shape
            features = patch_tokens.view(B, feature_h, feature_w, D).permute(0, 3, 1, 2)
        
        return features
    
    def forward(self, x):
        # x shape: [B, num_slices, 3, 224, 224]
        B, D, C, H, W = x.shape
        
        # 重塑为 [B*D, 3, 224, 224] 以便批量处理所有切片
        x_reshaped = x.view(B * D, C, H, W)
        
        # 批量提取所有切片的DINOv3特征
        dinov3_features = self.extract_2d_features(x_reshaped)  # [B*D, feature_dim, feat_h, feat_w]
        
        # 通过特征降维
        features_2d = self.feature_reducer(dinov3_features)  # [B*D, 128, feat_h, feat_w]
        
        # 上采样到合适尺寸
        feat_size = 56
        features_2d = F.interpolate(features_2d, size=(feat_size, feat_size), mode='bilinear', align_corners=False)
        
        # 重塑回3D格式: [B, D, 128, feat_h, feat_w] -> [B, 128, D, feat_h, feat_w]
        features_3d = features_2d.view(B, D, 128, feat_size, feat_size)
        features_3d = features_3d.permute(0, 2, 1, 3, 4)  # [B, 128, D, feat_h, feat_w]
        
        # 简单的3D卷积处理
        x = self.conv3d_1(features_3d)  # [B, 64, D, feat_h, feat_w]
        x = self.conv3d_2(x)           # [B, 32, D, feat_h, feat_w]
        
        # 最终输出
        output = self.final_conv(x)    # [B, 1, D, feat_h, feat_w]
        
        # 上采样到目标尺寸
        output = F.interpolate(output, size=(D, 224, 224), mode='trilinear', align_corners=False)
        
        return output

def dice_loss_3d(pred, target, smooth=1.):
    """3D Dice loss"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def calculate_dice_score_3d(pred, target, threshold=0.5, smooth=1e-6):
    """计算3D Dice系数"""
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    
    intersection = (pred & target).float().sum((1, 2, 3, 4))
    union = pred.float().sum((1, 2, 3, 4)) + target.float().sum((1, 2, 3, 4))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_iou_3d(pred, target, threshold=0.5):
    """计算3D IoU"""
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    
    intersection = (pred & target).float().sum((1, 2, 3, 4))
    union = (pred | target).float().sum((1, 2, 3, 4))
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

# 简化的指标计算函数（减少内存使用）
def calculate_basic_metrics_3d(pred_np, target_np):
    """计算基本的3D分割指标（内存优化版本）"""
    metrics = {}
    
    # 确保是二值化的
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    target_binary = (target_np > 0.5).astype(np.uint8)
    
    # Dice系数
    intersection = np.sum(pred_binary * target_binary)
    dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-6)
    metrics['dice'] = dice
    
    # IoU (Jaccard Index)
    union = np.sum((pred_binary + target_binary) > 0)
    iou = intersection / (union + 1e-6)
    metrics['iou'] = iou
    
    # 灵敏度 (Sensitivity/Recall)
    tp = intersection
    fn = np.sum(target_binary) - tp
    sensitivity = tp / (tp + fn + 1e-6)
    metrics['sensitivity'] = sensitivity
    
    # 精确度 (Precision)
    fp = np.sum(pred_binary) - tp
    precision = tp / (tp + fp + 1e-6)
    metrics['precision'] = precision
    
    # F1分数
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-6)
    metrics['f1_score'] = f1
    
    # 体积相关指标
    pred_volume = np.sum(pred_binary)
    target_volume = np.sum(target_binary)
    metrics['pred_volume_voxels'] = pred_volume
    metrics['target_volume_voxels'] = target_volume
    metrics['volume_difference_voxels'] = abs(pred_volume - target_volume)
    
    return metrics

def resize_prediction_to_original_simple(prediction, original_shape):
    """简化版本的resize，减少内存使用"""
    # prediction shape: [D, H, W] where D=num_slices, H=W=224
    # original_shape: [H, W, D]
    
    target_h, target_w, target_d = original_shape
    pred_d, pred_h, pred_w = prediction.shape
    
    # 简单的最近邻插值
    pred_resized = np.zeros(original_shape, dtype=np.float32)
    
    for z in range(target_d):
        # 找到对应的预测切片索引
        z_pred = int(z * pred_d / target_d)
        z_pred = min(z_pred, pred_d - 1)
        
        # 简单resize
        pred_slice = prediction[z_pred]
        
        # 使用numpy进行简单的双线性插值近似
        y_ratio = pred_h / target_h
        x_ratio = pred_w / target_w
        
        for y in range(target_h):
            for x in range(target_w):
                y_pred = min(int(y * y_ratio), pred_h - 1)
                x_pred = min(int(x * x_ratio), pred_w - 1)
                pred_resized[y, x, z] = pred_slice[y_pred, x_pred]
    
    return pred_resized

def save_prediction_as_nii(prediction, original_nii, output_path, threshold=0.5):
    """保存预测结果为.nii.gz文件"""
    try:
        # 应用阈值
        pred_binary = (prediction > threshold).astype(np.uint8)
        
        # 创建新的NIfTI图像
        pred_nii = nib.Nifti1Image(pred_binary, original_nii.affine, original_nii.header)
        
        # 保存
        nib.save(pred_nii, output_path)
        return True
    except Exception as e:
        print(f"Error saving prediction to {output_path}: {e}")
        return False

def evaluate_model_simple(model, dataset, device, output_dir, max_samples=10):
    """简化的模型评估函数（减少内存使用）"""
    model.eval()
    
    # 创建结果目录
    results_dir = os.path.join(output_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics = []
    saved_count = 0
    
    print(f"Evaluating model on {min(max_samples, len(dataset))} samples...")
    
    # 只评估指定数量的样本
    eval_indices = list(range(min(max_samples, len(dataset))))
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(eval_indices, desc="Evaluating")):
            try:
                # 获取预处理的数据
                volume_features, mask_tensor, _ = dataset[idx]
                volume_features = volume_features.unsqueeze(0).to(device)  # Add batch dimension
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, D, H, W]
                
                # 模型预测
                prediction = model(volume_features)  # [1, 1, D, H, W]
                prediction = torch.sigmoid(prediction).cpu().numpy()[0, 0]  # [D, H, W]
                
                # 清理GPU内存
                del volume_features
                torch.cuda.empty_cache()
                
                # 获取原始数据信息
                volume, mask, data_nii, label_nii, data_file, label_file = dataset.load_original_data(idx)
                
                if volume is None:
                    print(f"Skipping {data_file} due to loading error")
                    continue
                
                # 将预测结果resize回原始尺寸（使用简化版本）
                pred_resized = resize_prediction_to_original_simple(prediction, volume.shape)
                
                # 计算基本指标
                metrics = calculate_basic_metrics_3d(pred_resized, mask)
                metrics['filename'] = data_file
                metrics['original_shape'] = str(volume.shape)
                
                all_metrics.append(metrics)
                
                # 保存预测结果
                if saved_count < max_samples:
                    pred_filename = f"pred_{data_file}"
                    pred_path = os.path.join(results_dir, pred_filename)
                    
                    if save_prediction_as_nii(pred_resized, data_nii, pred_path):
                        saved_count += 1
                        print(f"Saved: {pred_filename} (Dice: {metrics['dice']:.4f})")
                
                # 清理内存
                del volume, mask, pred_resized, prediction
                gc.collect()
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # 保存结果
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        df_metrics.to_csv(csv_path, index=False)
        
        # 计算统计摘要
        summary = {}
        for col in ['dice', 'iou', 'sensitivity', 'precision', 'f1_score']:
            if col in df_metrics.columns:
                values = df_metrics[col]
                summary[f'{col}_mean'] = float(values.mean())
                summary[f'{col}_std'] = float(values.std())
                summary[f'{col}_median'] = float(values.median())
                summary[f'{col}_min'] = float(values.min())
                summary[f'{col}_max'] = float(values.max())
        
        # 保存摘要
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 打印结果
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for metric in ['dice', 'iou', 'sensitivity', 'precision']:
            if metric in df_metrics.columns:
                values = df_metrics[metric]
                print(f"{metric.upper()}: {values.mean():.4f} ± {values.std():.4f}")
        
        print(f"\nResults saved to: {csv_path}")
        print(f"Summary saved to: {summary_path}")
        print(f"Predictions saved: {saved_count}")
        
        return df_metrics, summary
    else:
        print("No valid results to save!")
        return None, None

def create_output_dir(model_path):
    """创建输出目录"""
    model_name = os.path.basename(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_3d_simple/{model_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    return output_dir

def train_model_3d_simple():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据路径
    train_data_dir = "/path/to/ncct_tr_niigz"
    train_label_dir = "/path/to/labelsTr_niigz"
    test_data_dir = "/path/to/ncct_ts_niigz"
    test_label_dir = "/path/to/labelsTs_niigz"
    
    model_path = "/path/to/weights/dinov3-vitb16-pretrain-lvd1689m"
    
    output_dir = create_output_dir(model_path + "_3D_Simple")
    print(f"Output directory: {output_dir}")
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    num_slices = 16
    train_dataset = SegmentationDataset3D(train_data_dir, train_label_dir, processor, is_train=True, num_slices=num_slices)
    test_dataset = SegmentationDataset3D(test_data_dir, test_label_dir, processor, is_train=False, num_slices=num_slices)
    
    if len(train_dataset) == 0:
        print("Error: No valid training data found!")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2) if len(test_dataset) > 0 else None
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 创建模型
    model = DINOv3Segmentation3D_Simple(model_path, num_classes=1, num_slices=num_slices).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练设置
    num_epochs = 50
    best_dice = 0
    debug_printed = False
    
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
    
    config_info = f"""Training Configuration (3D Simple):
Model: {os.path.basename(model_path)}
Architecture: 2D DINOv3 + Simple 3D Convs + Direct Upsampling
Device: {device}
Epochs: {num_epochs}
Batch Size: 4
Learning Rate: 1e-4
Number of Slices: {num_slices}
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
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                volumes, masks, _ = batch_data
            else:
                volumes, masks = batch_data
            
            volumes, masks = volumes.to(device), masks.to(device)
            
            # 确保masks的维度正确 [B, D, H, W] -> [B, 1, D, H, W]
            if len(masks.shape) == 4:
                masks = masks.unsqueeze(1)
            
            optimizer.zero_grad()
            
            if not debug_printed:
                print(f"Volumes shape: {volumes.shape}")
                print(f"Masks shape: {masks.shape}")
                outputs = model(volumes)
                print(f"Model output shape: {outputs.shape}")
                debug_printed = True
                print("3D Simple Shape debugging completed. Continuing training...")
            else:
                outputs = model(volumes)
            
            # 计算损失
            bce_loss = criterion(outputs, masks)
            dice_loss_val = dice_loss_3d(outputs, masks)
            loss = bce_loss + dice_loss_val
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou_3d(outputs, masks).item()
            train_dice += calculate_dice_score_3d(outputs, masks).item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{calculate_iou_3d(outputs, masks).item():.4f}',
                'Dice': f'{calculate_dice_score_3d(outputs, masks).item():.4f}'
            })
            
            # 清理GPU内存
            torch.cuda.empty_cache()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        
        if test_loader is not None:
            with torch.no_grad():
                for batch_data in test_loader:
                    if len(batch_data) == 3:
                        volumes, masks, _ = batch_data
                    else:
                        volumes, masks = batch_data
                    
                    volumes, masks = volumes.to(device), masks.to(device)
                    
                    if len(masks.shape) == 4:
                        masks = masks.unsqueeze(1)
                    
                    outputs = model(volumes)
                    
                    bce_loss = criterion(outputs, masks)
                    dice_loss_val = dice_loss_3d(outputs, masks)
                    loss = bce_loss + dice_loss_val
                    
                    val_loss += loss.item()
                    val_iou += calculate_iou_3d(outputs, masks).item()
                    val_dice += calculate_dice_score_3d(outputs, masks).item()
                    
                    # 清理GPU内存
                    torch.cuda.empty_cache()
        
        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        if test_loader is not None:
            avg_val_loss = val_loss / len(test_loader)
            avg_val_iou = val_iou / len(test_loader)
            avg_val_dice = val_dice / len(test_loader)
        else:
            avg_val_loss = avg_train_loss
            avg_val_iou = avg_train_iou
            avg_val_dice = avg_train_dice
        
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
        if test_loader is not None:
            print(f'Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}')
        
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
            print(f'New best model saved with Dice: {best_dice:.4f}')
        
        print('-' * 50)
    
    # 保存训练历史和绘制曲线
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_history['epoch'], train_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(train_history['epoch'], train_history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (3D Simple)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_history['epoch'], train_history['train_iou'], 'b-', label='Train IoU')
    plt.plot(train_history['epoch'], train_history['val_iou'], 'r-', label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU (3D Simple)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_history['epoch'], train_history['train_dice'], 'b-', label='Train Dice')
    plt.plot(train_history['epoch'], train_history['val_dice'], 'r-', label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Training and Validation Dice (3D Simple)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_3d_simple.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 简化的模型评估
    print("\n" + "="*60)
    print("LOADING BEST MODEL FOR EVALUATION")
    print("="*60)
    
    try:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoints', 'best_model.pth')))
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        # 在测试集上评估（只评估前10个样本）
        if len(test_dataset) > 0:
            print("Evaluating on test dataset (max 500 samples)...")
            test_metrics, test_summary = evaluate_model_simple(
                model, test_dataset, device, output_dir, max_samples=50
            )
        
        # 在训练集上评估（只评估前5个样本）
        print("Evaluating on training dataset (first 5 samples)...")
        train_eval_dir = os.path.join(output_dir, "train_evaluation")
        os.makedirs(train_eval_dir, exist_ok=True)
        train_metrics, train_summary = evaluate_model_simple(
            model, train_dataset, device, train_eval_dir, max_samples=1
        )
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Skipping detailed evaluation to avoid memory issues.")
    
    return output_dir

if __name__ == "__main__":
    print("Starting 3D training with DINOv3 (Simple Version - Memory Optimized)...")
    output_dir = train_model_3d_simple()
    if output_dir:
        print(f"\nAll results saved to: {output_dir}")
    else:
        print("Training failed due to data loading issues.")