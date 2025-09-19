import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import util as util
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.color import rgb2hsv
from PIL import Image


# 定义感知损失（Perceptual Loss）
'''
class PerceptualLoss(nn.Module):
    def __init__(self, device, layers_weights):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval().to(device)
        self.blocks = nn.ModuleList()
        self.layers_weights = layers_weights
        prev_idx = 0
        for idx in layers_weights.keys():
            block = nn.Sequential(*list(vgg.children())[prev_idx:idx])
            self.blocks.append(block)
            prev_idx = idx
        for param in self.blocks.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        for block, (layer_idx, weight) in zip(self.blocks, self.layers_weights.items()):
            x = block(x)
            y = block(y)
            loss += weight * torch.mean(torch.abs(x - y))
        return loss
'''

class PerceptualLoss(nn.Module):
    def __init__(self,device,):
        super(PerceptualLoss, self).__init__()

        vgg = models.vgg19(pretrained=True).to(device)
        loss_network = nn.Sequential(*list(vgg.features)[:6]).eval() #35
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, B, I):
        perception_loss = self.l1_loss(self.loss_network(B), self.loss_network(I))
        return perception_loss

class Pyramid(nn.Module):
    """Pyramid Loss"""
    def __init__(self, num_levels=3, pyr_mode='gau'):
        super(Pyramid, self).__init__()
        self.num_levels = num_levels
        self.pyr_mode = pyr_mode
        assert self.pyr_mode == 'gau' or self.pyr_mode == 'lap'

    def forward(self, x, local_rank):
        B, C, H, W = x.shape
        gauss_kernel = util.gauss_kernel(size=5, device=local_rank, channels=C)
        if self.pyr_mode == 'gau':
            pyr_x = util.gau_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
        else:
            pyr_x = util.lap_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
        return pyr_x

# IBS Loss
class IBSLoss(nn.Module):
    def __init__(self,device):
        super(IBSLoss, self).__init__()
        #self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        # 使用VGG19的前几个层作为感知损失，偏重低层特征
        '''self.percep_loss = PerceptualLoss(device, layers_weights={
            2: 1.0,  # relu1_1
            7: 0.5,  # relu2_1
            12: 0.2  # relu3_1
        })'''
        #self.percep_loss = PerceptualLoss(device)

    def forward(self, B, I):
        l1 = self.l1_loss(B, I)
        #perceptual = self.percep_loss(B, I)
        return l1 #+ 0.05*perceptual

class ColorConstancyLoss(nn.Module):
    def __init__(self, delta=0.1):
        super(ColorConstancyLoss, self).__init__()
        self.delta = delta
        self.l1_loss = nn.L1Loss()  # 按照公式累加后除以K

    def forward(self, I, J):
        """
        I, J: tensors of shape (B, 3, H, W)
        Returns: scalar loss
        """
        assert I.shape == J.shape and I.shape[1] == 3, "Expect input of shape (B, 3, H, W)"
        channel_pairs = [(0, 1), (0, 2), (1, 2)]
        loss = 0.0

        for p, q in channel_pairs:
            ratio_I = (I[:, p:p+1] + self.delta) / (I[:, q:q+1] + self.delta)
            ratio_J = (J[:, p:p+1] + self.delta) / (J[:, q:q+1] + self.delta)
            loss += self.l1_loss(ratio_I, ratio_J)

        return loss / len(channel_pairs)
def exclusion_loss(B, R, local_rank, levels=3):
    Pyr = Pyramid(num_levels=3, pyr_mode='lap')
    B_grad = Pyr(B, local_rank)
    R_grad = Pyr(R, local_rank)
    loss = 0
    for l in range(levels):
        # normalize gradients

        # element-wise product
        # 对每个通道除以该通道上的最大值（在每个样本内）
        #sum_vals = B_grad[l].sum(dim=(1, 2, 3), keepdim=True)  # shape: [B, 1, 1, 1]
        #B_norm = B_grad[l] / (sum_vals + 1e-8)
        #sum_vals = R_grad[l].sum(dim=(1, 2, 3), keepdim=True)  # shape: [B, 1, 1, 1]
        #R_norm = R_grad[l] / (sum_vals + 1e-8)
        #alpha = torch.mean(torch.abs(B_grad[l])) / (torch.mean(torch.abs(R_grad[l])) + 1e-6)
        mse_exl= (torch.sigmoid(B_grad[l]) * torch.sigmoid(R_grad[l]))**2
        mse_exl = mse_exl.mean(dim=(1, 2, 3))
        mse_exl = mse_exl.mean()
        #loss += torch.norm(B_grad[l] * R_grad[l], p='fro')/levels
        loss += mse_exl / levels

    return loss

def cap_loss(B):
    b,c,h,w = B.shape
    hsv_list = []
    for i in range(b):
        # Step 1: 取出第 i 个样本并转为 numpy 的 [H, W, 3]
        rgb_np = B[i].detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

        # Step 2: RGB → HSV (matplotlib.colors.rgb_to_hsv expects [H, W, 3])
        hsv_np = rgb2hsv(rgb_np)  # 输出仍是 [H, W, 3]
        hsv_torch = torch.from_numpy(hsv_np.transpose(2, 0, 1))  # [3, H, W]

        hsv_list.append(hsv_torch)

    hsv = torch.stack(hsv_list, dim=0).to(B.device).float()
    cap_prior = hsv[:, 1, :, :] - hsv[:, 2, :, :]
    #Image.fromarray((hsv[:, 1, :, :].squeeze()*255).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()).save('/data/cyt2/difussionRR/save_RR/s.png')
    #Image.fromarray((hsv[:, 2, :, :].squeeze()*255).clamp(0, 255).to(torch.uint8).squeeze().detach().cpu().numpy()).save('/data/cyt2/difussionRR/save_RR/v.png')
    #Image.fromarray((hsv[:, 0, :, :].squeeze() * 255).clamp(0, 255).to(torch.uint8).squeeze().detach().cpu().numpy()).save('/data/cyt2/difussionRR/save_RR/h.png')
    l1_loss = nn.L1Loss()
    cap_value = l1_loss(cap_prior, torch.zeros_like(cap_prior))
    return cap_value

def lspa_loss(O, R, alpha=1.0):
    """
    O, R: tensors of shape (B, C, H, W)
    alpha: weighting factor for psi term
    """
    B, C, H, W = O.shape
    device = O.device

    # Define spatial neighbor offsets for φ(i) and ψ(i)
    phi_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # 4-neighbor
    psi_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]   # diagonal 4-neighbor (8 total with φ)

    def spatial_l1_diff(x, y, offsets):
        total_loss = 0
        for dy, dx in offsets:
            # shift y
            shifted_y = F.pad(y, (1,1,1,1), mode='reflect')
            shifted_y = shifted_y[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]

            # compute l1 norm
            dist_x = (x - shifted_y).abs().sum(dim=1, keepdim=True)  # shape: (B,1,H,W)
            total_loss += dist_x
        return total_loss

    # Compute pairwise L1 diffs in φ(i) and ψ(i)
    O_phi = spatial_l1_diff(O, O, phi_offsets)
    R_phi = spatial_l1_diff(R, R, phi_offsets)

    O_psi = spatial_l1_diff(O, O, psi_offsets)
    R_psi = spatial_l1_diff(R, R, psi_offsets)

    # Compute squared difference of the L1 distances
    loss_phi = (O_phi - R_phi).pow(2).mean()
    loss_psi = (O_psi - R_psi).pow(2).mean()

    return loss_phi + alpha * loss_psi


class PhaseLoss(nn.Module):
    def __init__(self, loss_weight=1, epsilon=1e-8):
        """
        初始化 PhaseLoss 类。

        :param epsilon: 防止除零或其他数值不稳定的小常数
        """
        super(PhaseLoss, self).__init__()
        self.epsilon = epsilon
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        # 转换为灰度图，确保张量在 0-1 范围内
        gray_img1 = torch.mean(img1, dim=1, keepdim=True)
        gray_img2 = torch.mean(img2, dim=1, keepdim=True)

        # 计算 FFT，加上 epsilon 提高数值稳定性
        fft_img1 = torch.fft.fft2(gray_img1 + self.epsilon)
        fft_img2 = torch.fft.fft2(gray_img2 + self.epsilon)

        # 获取相位信息并归一化到 [-π, π]
        phase_img1 = torch.angle(fft_img1)
        phase_img2 = torch.angle(fft_img2)

        # 计算相位差的 MSE 损失
        loss = F.mse_loss(phase_img1, phase_img2)

        return self.loss_weight * loss

'''
def exclusion_loss(img1, img2, level=1):
    gradx_loss = []
    grady_loss = []

    for _ in range(level):
        gradx1, grady1 = compute_gradient(img1)
        gradx2, grady2 = compute_gradient(img2)

        alphax = 2.0 * torch.mean(torch.abs(gradx1)) / (torch.mean(torch.abs(gradx2)) + 1e-6)
        alphay = 2.0 * torch.mean(torch.abs(grady1)) / (torch.mean(torch.abs(grady2)) + 1e-6)

        gradx1_s = torch.sigmoid(gradx1) * 2 - 1
        grady1_s = torch.sigmoid(grady1) * 2 - 1
        gradx2_s = torch.sigmoid(gradx2 * alphax) * 2 - 1
        grady2_s = torch.sigmoid(grady2 * alphay) * 2 - 1

        gx_loss = torch.mean((gradx1_s ** 2) * (gradx2_s ** 2)) ** 0.25
        gy_loss = torch.mean((grady1_s ** 2) * (grady2_s ** 2)) ** 0.25

        gradx_loss.append(gx_loss)
        grady_loss.append(gy_loss)

        img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
        img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)

    return gradx_loss, grady_loss
'''



def compute_gradient(img):
    gradx = img[:, 1:, :, :]-img[:, :-1, :, :]
    grady = img[:, :, 1:, :]-img[:, :, :-1, :]
    return gradx, grady
def cross_consistent_loss(B_t, R_t, I):
    l2_loss = nn.MSELoss()
    I_pred = B_t + R_t
    #I_pred = I_pred.clamp(min=0, max=1)
    loss = l2_loss(I_pred, I)
    return loss

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d


class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        count_h =  (x.size()[-2]-1) * x.size()[-2]
        count_w = x.size()[-1] * (x.size()[-1] - 1)
        h_tv = torch.pow((x[1:,:]-x[:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,1:]-x[:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class TVL1(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVL1, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        count_h = self._tensor_size(x[..., 1:, :])
        count_w = self._tensor_size(x[..., :, 1:])

        h_tv = torch.abs((x[..., 1:, :] - x[..., :h_x - 1, :])).sum()
        w_tv = torch.abs((x[..., :, 1:] - x[..., :, :w_x - 1])).sum()
        # print("h,w:", h_tv, w_tv)
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size
    def _tensor_size(self, t):
        return t.size()[-3] * t.size()[-2] * t.size()[-1]

class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3