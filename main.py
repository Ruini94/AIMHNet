import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from net_HourGlass import Net
from misc import AvgMeter
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
from MyDataset import MyDataset
# from ptflops import get_model_complexity_info
from pytorch_msssim import ssim,ms_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True


# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 


parser = argparse.ArgumentParser()
parser.add_argument("--root", default=r"/rain_dataset/Rain1200/")
parser.add_argument("--name", default="Rain1200")
parser.add_argument("--train_batch_size", default=8)
parser.add_argument("--model_path", default="best_models")

# parser.add_argument("--device_ids", default=[0,1,2,3])  # 768/6=128  1536/6=256

parser.add_argument("--iter_num", default=100)

parser.add_argument("--last_iter", default=0)
parser.add_argument("--lr", default=5e-4)  # 1e-4
parser.add_argument("--lr_decay", default=0.9)
parser.add_argument("--weight_decay", default=0)
parser.add_argument("--momentum", default=0.9)
parser.add_argument("--resume_snapshot")
parser.add_argument("--val_freq", default=1)  # validate frequency
parser.add_argument("--snapshot_epochs", default=5)  # save model frequency


args = parser.parse_args()
print(args)
os.makedirs(args.model_path, exist_ok=True)


train_set = MyDataset(root=args.root, name=args.name, cropSize=256, mode="train")
train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=1, shuffle=True, drop_last=True)
test1_set = MyDataset(root=args.root, name=args.name, cropSize=256, mode="test")
test1_loader = DataLoader(test1_set, batch_size=4, shuffle=False)


GRAD_CLIP = 1.0
criterion = nn.L1Loss()

log_path = os.path.join("logs", str(datetime.datetime.now().timestamp()) + '.txt')

net = Net().cuda() 
print(net)

net.train()

optimizer = optim.Adam(
    [  # name 'mean' param.shape 1,3,1,1  'std' param.shape 1,3,1,1  conv1.0.weight param.shape 32,3,4,4
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args.lr},  # 如果参数名字里面有 bias 且这个参数的 requires_grad 是 True  学习率乘以 2
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args.lr, 'weight_decay': args.weight_decay}  # 如果参数名字里面没有 bias 则 学习率会衰减
    ])


os.makedirs("logs", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
open(log_path, 'w').write(str(args) + '\n\n')


def save_checkpoint(model, epoch, psnr, avg_loss):
    model_out_path = os.path.join(args.model_path,
                                  "{}_epoch_{}_PSNR_{:.4f}_loss_{:.6f}.pth".format(args.name, epoch, psnr, avg_loss))

    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(args.model_path))


def validate(epoch):
    print('validating...')
    net.eval()
    with torch.no_grad():
        rain_img, _ = next(iter(test1_loader))
        rain_img = rain_img.cuda()
        res = net(rain_img)
        img_sample = torch.cat((rain_img, res), 0)
        save_image(img_sample, "images/%s.png" % epoch, nrow=4, normalize=True)

    net.train()


best_psnr = 0.0
if __name__ == '__main__':
    for epoch in tqdm(range(args.iter_num + 1)):
        train_loss_record = AvgMeter()
        l1_loss_record = AvgMeter()
        struc_loss_record = AvgMeter()
        for i, data in enumerate(train_loader):  # 这个是总迭代次数 args['iter_num']
            optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(epoch) / args.iter_num
                                                             ) ** args.lr_decay
            optimizer.param_groups[1]['lr'] = args.lr * (1 - float(epoch) / args.iter_num
                                                         ) ** args.lr_decay

            inputs, gts = data
            batch_size = inputs.size(0)
            inputs = inputs.cuda()
            gts = gts.cuda()

            optimizer.zero_grad()
            result = net(inputs)  # 4,3,256,256

            loss_net = criterion(result, gts)  # 这两个都是 L1Loss
            loss_struc = 2-ssim(result, gts, data_range=1.0)-ms_ssim(result, gts, data_range=1.0)
            loss = loss_net + loss_struc

            loss.backward()
            
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            if total_norm > GRAD_CLIP * 1000:
                print('total_norm', total_norm)
                
            optimizer.step()
            
            
            train_loss_record.update(loss.data, batch_size)
            l1_loss_record.update(loss_net.data, batch_size)
            struc_loss_record.update(loss_struc.data, batch_size)

            log = '[iter %d], [train loss %.5f], [l1 loss %.5f], [structural loss %.5f] [lr %.13f]' % \
                  (epoch, train_loss_record.avg, l1_loss_record.avg, struc_loss_record.avg, optimizer.param_groups[1]['lr'])

            print(log, flush=True)
            open(log_path, 'a').write(log + '\n')
        
        
        psnr = 0
        for i in range(test1_set.__len__()):
            val_data, val_label = test1_set[i]
            val_data = val_data.unsqueeze(0).cuda()
            val_label_numpy = val_label.numpy()
            with torch.no_grad():
                val_out = net(val_data)
            val_out = val_out.squeeze(0).cpu().detach().numpy()
            psnr += compare_psnr(val_out, val_label_numpy, data_range=1.0)

        psnr = psnr / (i + 1)
        loss_avg = 0
        if psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(net, epoch, best_psnr, loss_avg)
              
        if (epoch + 1) % args.val_freq == 0:
            validate(epoch)

       
