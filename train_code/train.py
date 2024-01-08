from architecture import *
from utils import *
from dataset import Dataset
import paths
import torch.utils.data as tud
import torch
from torch.autograd import Variable
import os
from option import opt
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from Discriminator import Discriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
if not os.path.exists(paths.model_path):
    os.makedirs(paths.model_path)

# load training data
file_paths, num_files = prepare_path(paths.train_path)
file_paths_val, num_files_val = prepare_path(paths.valid_path)

# model
model = model_generator(opt.method, opt.stage)
discriminator = Discriminator(opt.channel).cuda()

# tensor board
if opt.tb:
    tensorBoardPath = f"{opt.path_root}DUNTS/data/log/{opt.method}_{paths.model_name}"
    if not os.path.exists(tensorBoardPath):
        os.makedirs(tensorBoardPath)
    writer = SummaryWriter(log_dir=f"{tensorBoardPath}")

# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate)


bias_list = [i for i in range(opt.channel)]
criterion = nn.L1Loss()

if __name__ == "__main__":

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    dataset_train = Dataset(file_paths, num_files, bias_list, opt.sim)
    dataset_valid = Dataset(file_paths_val, num_files_val, bias_list, opt.sim)
    loader_train = tud.DataLoader(dataset_train, num_workers=8, batch_size=opt.batch_size, shuffle=True)
    loader_valid = tud.DataLoader(dataset_valid, num_workers=8, batch_size=opt.batch_size, shuffle=False)

    for epoch in range(opt.first_epoch, opt.max_epoch):
        epoch_loss_d = 0
        epoch_loss_g = 0

        gradient_dict = {}
        model.train()
        for i, (y, gt, mask) in enumerate(loader_train):
            y, gt, mask = Variable(y), Variable(gt), Variable(
                mask)
            y, gt, mask = y.cuda(), gt.cuda(), mask.cuda()

            # 步骤1: 训练判别器
            discriminator_optimizer.zero_grad()
            # 生成生成器的输出
            x = init_input(y, bias_list=bias_list)
            out = model(y, x, mask, bias_list=bias_list)
            out = torch.clamp(out, 0., 1.)
            # 将真实图像和生成的图像输入到判别器中
            real_outputs = discriminator(rearrange(gt, 'b c t h w -> b (c t) h w'))
            fake_outputs = discriminator(out.detach())  # 注意：detach以防止生成器被更新
            # 计算判别器损失
            d_loss_real = criterion(real_outputs, torch.ones_like(real_outputs))
            d_loss_fake = criterion(fake_outputs, torch.zeros_like(fake_outputs))
            total_d_loss = d_loss_real + d_loss_fake
            epoch_loss_d += total_d_loss.item()
            total_d_loss.backward()
            discriminator_optimizer.step()

            # 步骤2: 训练生成器
            optimizer.zero_grad()
            # 重新生成图像
            x = init_input(y, bias_list=bias_list)
            out = model(y, x, mask, bias_list=bias_list)
            out = torch.clamp(out, 0., 1.)
            # 输入生成器的生成图像到判别器中
            fake_outputs = discriminator(out)
            # 计算生成器损失（包括重建误差）
            reconstruction_loss = criterion(out, rearrange(gt, 'b c t h w -> b (c t) h w'))
            g_loss = reconstruction_loss + 0.7 * torch.mean(fake_outputs)  # 添加GAN损失
            epoch_loss_g += g_loss.item()
            g_loss.backward()
            optimizer.step()

        with open(f"{opt.path_root}DUNTS/data/gradient_log/{opt.method}_{paths.model_name}.txt", 'a') as f:
            f.write(f"Epoch: {epoch}\n")
            for name, gradient in gradient_dict.items():
                f.write(f"{name}: {gradient}\n")
        print(f"epoch {epoch + 1} / {opt.max_epoch}:"
              f"    d_loss = {epoch_loss_d / len(dataset_train):.7f}"
              f"    g_loss = {epoch_loss_g / len(dataset_train):.7f}")
        if opt.tb:
            writer.add_image('mea', y[0, :, :], global_step=epoch, dataformats='HW')
            train_show_gan(gt, out, 'train', epoch, writer)
            writer.add_scalar('g_loss', epoch_loss_g / len(dataset_train), global_step=epoch)
            writer.add_scalar('d_loss', epoch_loss_d / len(dataset_train), global_step=epoch)

        if epoch % 20 == 0 and epoch != 0:
            torch.save(model, os.path.join(paths.model_path, 'model_%03d.pth' % (epoch + 1)))
            torch.save(discriminator, os.path.join(paths.discriminator_path, 'model_%03d.pth' % (epoch + 1)))

        psnr_list = []
        ssim_list = []
        model.eval()
        with torch.no_grad():
            for i, (y, gt, mask) in enumerate(loader_valid):
                y, gt, mask = Variable(y), Variable(gt), Variable(
                    mask)
                y, gt, mask = y.cuda(), gt.cuda(), mask.cuda()

                x = init_input(y, bias_list=bias_list)
                out = model(y, x, mask, bias_list=bias_list)
                out = rearrange(out, 'b (c t) h w -> b c t h w', c=opt.channel)
                out = torch.clamp(out, 0., 1.)

                psnr_list = psnr_list + PSNR_valid(out, gt)
                ssim_list = ssim_list + SSIM_valid(out, gt)

            psnr_valid = np.average(psnr_list)
            ssim_valid = np.average(ssim_list)
            print(f"PSNR = {psnr_valid:.4f}, SSIM = {ssim_valid:.4f}")
            if opt.tb:
                train_show(out, gt, 'valid', epoch, writer)
                writer.add_scalar('psnr', psnr_valid, global_step=epoch)
                writer.add_scalar('ssim', ssim_valid, global_step=epoch)
