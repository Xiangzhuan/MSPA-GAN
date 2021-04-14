import numpy as np
#import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap
from skimage.measure import compare_ssim as SSIM
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager import TestDataset
from utils import gpu_manage, save_image, heatmap
from models.gen.SPANet import Generator
import torch.nn as nn


def predict(config, args):
    gpu_manage(args)
    dataset = TestDataset(args.test_dir, args.txtdir,config.in_ch, config.out_ch)
    data_loader = DataLoader(
        dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)

    loss = []
    
    avg_mse=0
    avg_psnr=0
    avg_ssim=0
    mseloss=nn.L1Loss()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch[0])
            filename = batch[1][0]
            if args.cuda:
                x = x.cuda()

            att, out = gen(x)

            h = 1
            w = 3
            c = 3
            p = config.width

            allim = np.zeros((h, w, c, p, p))
            x_ = x.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            att_ = att.cpu().numpy()[0] * 255
            print("attention size:",att.size())
            heat_att = heatmap(att_.astype('uint8'))

            # 计算损失值 并存储
            criterionL1 = nn.L1Loss()
            loss_g_l1 = criterionL1(out, x) * config.lamb
            loss_g_l1 = loss_g_l1.cpu().numpy().item()
            loss.append([filename,loss_g_l1])

            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = heat_att
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*p, w*p, c))
            
            #计算psnr ssim mse
            mse=mseloss(out,x)
            psnr=10*np.log10(1/mse.item())
            img1 = np.tensordot(out.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
            img2 = np.tensordot(x.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
            ssim = SSIM(img1, img2)
            avg_mse += mse.item()
            avg_psnr += psnr
            avg_ssim += ssim
            
            save_image(args.out_dir, allim, i, 1, filename=filename)
            
    avg_mse = avg_mse / len(data_loader)
    avg_psnr = avg_psnr / len(data_loader)
    avg_ssim = avg_ssim / len(data_loader)
    return loss,avg_mse,avg_psnr,avg_ssim


class Args():
    txtdir="D:\\pythonRepos\\SpA-GAN_for_cloud_removal-master\\data\\RICE_DATASET\\RICE2\\test_list.txt"
    test_dir = "D:\\pythonRepos\\SpA-GAN_for_cloud_removal-master\\data\\RICE_DATASET\\RICE2"
    out_dir = "c:/users/ssk/desktop/results/75"
    pretrained = "D:\\pythonRepos\\SPA-GAN_loss_modified\\results\\lamb=75 15\\models\\gen_model_epoch_90.pth"
    cuda = True
    gpu_ids = [0]
    manualSeed = 2


if __name__ == '__main__':
    '''parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=False)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)

    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.load(f)
    config = AttrMap(config)'''
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f)
    config = AttrMap(config)
    args = Args()

    best_img = []
    loss1,avg_mse,avg_psnr,avg_ssim= predict(config, args)
    
    for i in range(20):
        minvalue = min(loss1,key=lambda x: x[1])
        index = loss1.index(minvalue)
        best_img.append(loss1[index])
        loss1.remove(minvalue)
    print("最好的20张图片是{}".format(best_img))
    print("avg_mse is {}, avg_psnr is {}, avg_ssim is {}".format(avg_mse,avg_psnr,avg_ssim))
    
