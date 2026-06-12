import os.path
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from Reader import CombinedDataLoader
from PRSLoss import *
import PRSNet
from datetime import datetime

num_epochs = 20
device_name = "cuda" if torch.cuda.is_available() else "cpu"


def train(num_epochs, data_iter, model, optimizer, loss_func, writer):
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (voxels, points, idx_maps, pts_maps, max_min) in enumerate(data_iter):
            optimizer.zero_grad()
            planes, quaternions = model(voxels)
            loss = loss_func(planes, quaternions, points, idx_maps, pts_maps, max_min)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # ── TensorBoard：每 batch 记录 loss ──
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        avg_loss = epoch_loss / max(len(data_iter), 1)

        # ── TensorBoard：每 epoch 记录平均 loss ──
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

        # ── TensorBoard：记录学习率 ──
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # ── TensorBoard：记录参数直方图（每 5 个 epoch） ──
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        print(f"┌─ Epoch {epoch:3d} avg loss: {avg_loss:.6f}")
        print(f"└─ lr: {optimizer.param_groups[0]['lr']:.2e}\n")


if __name__ == "__main__":
    # ── TensorBoard 日志目录 ──
    log_dir = f"./runs/prs-net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f" TensorBoard 日志目录: {log_dir}")
    print(f"   启动: tensorboard --logdir={log_dir}")

    # ── 数据 ──
    dataloader = CombinedDataLoader(
        intermediate_data_dir='./shapenet_intermediate',
        batch_size=32,
        num_workers=0,
        shuffle=True,
        device=device_name
    )

    # ── 模型 ──
    model = PRSNet.PRSNet().to(device_name)
    LossFunc = SymmetryLoss().to(device_name)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # ── 记录模型图（可选，需要 dummy input） ──
    dummy_voxel = torch.randn(1, 1, 32, 32, 32).to(device_name)
    writer.add_graph(model, dummy_voxel)

    # ── 训练 ──
    train(num_epochs, dataloader, model, optimizer, LossFunc, writer)

    # ── 保存 ──
    torch.save(model.state_dict(), './prs-net.pth')
    print(f"✅ 模型已保存: ./prs-net.pth")

    writer.close()
    print(f"📊 TensorBoard 已关闭")
    print(f"   查看: tensorboard --logdir={log_dir}")