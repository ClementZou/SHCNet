from utils import *
import torch
import paths
from torch.autograd import Variable
import os
from option import opt
from dataset import Dataset
import torch.utils.data as tud
from einops import rearrange

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
opt.isTrain = False


file_paths, num_files = prepare_path(paths.valid_path)
bias_list = [i for i in range(opt.channel)]

pretrained_model_base_path = os.path.join(paths.model_path, f"model_{opt.test_epoch:03d}.pth")
model = torch.load(pretrained_model_base_path)
model = model.eval()

if __name__ == "__main__":
    dataset_test = Dataset(file_paths, num_files, bias_list, opt.sim)
    loader = tud.DataLoader(dataset_test, num_workers=8, batch_size=1, shuffle=False)
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for i, (y, gt, mask) in enumerate(loader):
            y, gt,  mask = Variable(y), Variable(gt), Variable(
                mask)
            y, gt, mask = y.cuda(), gt.cuda(), mask.cuda()

            x = init_input(y, bias_list=bias_list)
            out, _ = model(y, x, mask, bias_list=bias_list)
            out = rearrange(out, 'b (c t) h w -> b c t h w', c=opt.channel)
            out = torch.clamp(out, 0., 1.)

            PSNR_temp = PSNR_valid(out, gt)
            SSIM_temp = SSIM_valid(out, gt)
            psnr_list = psnr_list + PSNR_temp
            ssim_list = ssim_list + SSIM_temp
            test_save(y, out, gt, PSNR_temp, SSIM_temp, i)
            if i % 20 == 0:
                print(f"{i} / {len(dataset_test)}")
        psnr_valid = np.average(psnr_list)
        ssim_valid = np.average(ssim_list)
        log_save_sim(psnr_valid, ssim_valid)