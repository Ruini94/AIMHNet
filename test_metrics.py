import os
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from net_HourGlass import Net

from tqdm import tqdm
import lpips
import colour
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


model = Net().cuda()
# model = nn.DataParallel(model, [0])

name = "Rain1200"
ckp_path = r"best_models/Rain1200_epoch.pth"
pre = torch.load(ckp_path)
# pre = pre.module.state_dict()
model.load_state_dict(pre, strict=True)

path1 = r"Rain1200/testA/"
print(path1)
path2 = r"AIMHNet/Rain1200/Rain1200_test_outputs/"
targetPath = r"Rain1200/testB/"

os.makedirs(path2, exist_ok=True)

for x in sorted(os.listdir(path1)):
    img1 = os.path.join(path1, x)


testA_list = [os.path.join(path1, x) for x in sorted(os.listdir(path1))]
testB_list = [os.path.join(path2, x) for x in sorted(os.listdir(path1))]

target_list = [os.path.join(targetPath, x) for x in sorted(os.listdir(targetPath))]
transf = transforms.ToTensor()
resize_ = transforms.Resize((240, 480))

if __name__ == "__main__":
    with torch.no_grad():
        model.eval()
        for i in range(testA_list.__len__()):
            img = Image.open(testA_list[i])
            if len(img.split()) != 3:
                img = img.convert("RGB")

            input = transf(img).unsqueeze(0).cuda()
            output = model(input)
            output = np.uint8(output.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)*255)
            Image.fromarray(output).save(testB_list[i])

    model.cpu()
    
    psnr_score = 0.0
    ssim_score = 0.0
    lpips_score = 0.0
    deltae_score = 0.0
    
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    
    for i in tqdm(range(testA_list.__len__())):
        img = np.array(transf(Image.open(testB_list[i])))
        if name == "RainCityscapes":
            target = np.array(transf(Image.open(target_list[i // 36])))
        else:
            target = np.array(transf(Image.open(target_list[i])))
        
        psnr_test = compare_psnr(img, target, data_range=1)
        ssim_test = compare_ssim(img.transpose(1, 2, 0), target.transpose(1, 2, 0), multichannel=True)

        img_cv = cv2.imread(testB_list[i])
        if name == "RainCityscapes":
            target_cv = cv2.imread(target_list[i // 36])
        else:
            target_cv = cv2.imread(target_list[i])
            
        img_lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_cv, cv2.COLOR_BGR2LAB)

        deltae_test = np.mean(colour.delta_E(img_lab, target_lab))

        img_t = resize_(torch.from_numpy(img)).unsqueeze(0).cuda()
        target_t = resize_(torch.from_numpy(target)).unsqueeze(0).cuda()
        
        lpips_test = loss_fn_alex(img_t, target_t)

        psnr_score = psnr_score + psnr_test
        ssim_score = ssim_score + ssim_test
        lpips_score = lpips_score + lpips_test
        deltae_score = deltae_score + deltae_test

    psnr_score = psnr_score / (testA_list.__len__())
    ssim_score = ssim_score / (testA_list.__len__())
    lpips_score = lpips_score / (testA_list.__len__())
    deltae_score = deltae_score / (testA_list.__len__())
    print("psnr score is: %.4f, ssim score is %.4f, lpips score is %.4f, deltaE score is %.4f" % (psnr_score,
                                                                                                  ssim_score,
                                                                                                  lpips_score,
                                                                                                  deltae_score))
