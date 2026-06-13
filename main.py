import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from Reader import CombinedDataLoader, CombinedDataset
from PRSLoss import *
import PRSNet

# ==================== 配置 ====================
num_epochs = 100
batch_size = 32
device_name = "cuda" if torch.cuda.is_available() else "cpu"
val_split = 0.15  # 15% 验证集
best_model_path = './prs-net-best.pth'
last_model_path = './prs-net-last.pth'
patience = 15  # 早停：验证 loss 不降超过 15 轮就停


def evaluate(model, dataloader, loss_func):
    """在验证集上计算平均 loss"""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for voxels, points, idx_maps, pts_maps, max_min in dataloader:
            planes, quaternions = model(voxels)
            loss = loss_func(planes, quaternions, points, idx_maps, pts_maps, max_min)
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def train(num_epochs, train_loader, val_loader, model, optimizer, scheduler,
          loss_func, writer, save_path, patience):
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # ==================== 训练阶段 ====================
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for voxels, points, idx_maps, pts_maps, max_min in train_loader:
            optimizer.zero_grad()
            planes, quaternions = model(voxels)
            loss = loss_func(planes, quaternions, points, idx_maps, pts_maps, max_min)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        avg_train_loss = epoch_loss / max(batch_count, 1)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # ==================== 验证阶段 ====================
        avg_val_loss = evaluate(model, val_loader, loss_func)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # 学习率调度
        scheduler.step()

        # ==================== 打印 ====================
        is_best = avg_val_loss < best_val_loss
        flag = " 🏆 BEST" if is_best else ""
        print(f"┌─ Epoch {epoch:3d} | train: {avg_train_loss:.6f} | val: {avg_val_loss:.6f}{flag}")
        print(f"└─ lr: {optimizer.param_groups[0]['lr']:.2e}")

        # ==================== 参数直方图（每 10 epoch）====================
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # ==================== 保存最优模型 ====================
        if is_best:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss,
            }, save_path)
            print(f"    ✅ 最优模型已保存 → {save_path}")
        else:
            epochs_without_improvement += 1

        # ==================== 早停 ====================
        if epochs_without_improvement >= patience:
            print(f"\n⏹️  早停：验证 loss 连续 {patience} 轮未改善")
            break

    return best_val_loss


if __name__ == "__main__":
    # ==================== TensorBoard ====================
    log_dir = "./runs/prs-net"
    writer = SummaryWriter(log_dir=log_dir)
    print(f" TensorBoard: tensorboard --logdir={log_dir}\n")

    # ==================== 数据集划分 ====================
    full_dataset = CombinedDataset(
        intermediate_data_dir='./shapenet_intermediate',
        device=device_name  # 预加载在 CPU，DataLoader 会搬
    )

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size

    # 固定种子保证可复现
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"   总样本: {len(full_dataset)}")
    print(f"   训练集: {train_size}")
    print(f"   验证集: {val_size}\n")

    # ==================== DataLoader ====================
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device_name != 'cuda'),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_name != 'cuda'),
    )

    # ==================== 模型 ====================
    model = PRSNet.PRSNet().to(device_name)
    loss_func = PRSLoss().to(device_name)


    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    设备: {device_name}\n")

    # ==================== 优化器 + 调度器 ====================
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ==================== 记录计算图 ====================
    dummy_voxel = torch.randn(1, 1, 32, 32, 32).to(device_name)
    writer.add_graph(model, dummy_voxel)

    # ==================== 训练 ====================
    best_loss = train(
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        writer=writer,
        save_path=best_model_path,
        patience=patience,
    )

    # ==================== 保存最后一轮 ====================
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': best_loss,
    }, last_model_path)

    print(f"\n 最后一轮模型: {last_model_path}")
    print(f" 最优模型:     {best_model_path}  (val_loss={best_loss:.6f})")

    writer.close()
    print(f"   TensorBoard 已关闭")
