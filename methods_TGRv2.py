#TGR_V2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
import scipy.stats as st
import copy
from utils import ROOT_PATH
from functools import partial
import copy
import pickle as pkl
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import params
from model import get_model
class BaseAttack(object):
    def __init__(self, attack_name, model_name, target):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        self.loss_flag = -1 if target else 1
        self.used_params = params(self.model_name)
        self.model = get_model(self.model_name)
        self.model.cuda()
        self.model.eval()
    def forward(self, *input): raise NotImplementedError
    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.mul_(std[:,None, None]).add_(mean[:,None,None])
        return inps
    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps = (inps - mean[:,None,None])/std[:,None,None]
        return inps
    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps)
        for i,filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image = unnorm_inps[i].permute([1,2,0])
            image[image<0] = 0
            image[image>1] = 1
            image = Image.fromarray((image.detach().cpu().numpy()*255).astype(np.uint8))
            image.save(save_path)
    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts
    def __call__(self, *input, **kwargs): return self.forward(*input, **kwargs)

class TGR_Innovative_MDITI(BaseAttack):
    def __init__(self, model_name, steps=10, epsilon=16/255, alpha=1.6/255, target=False, decay=1.0,
                 num_diverse=8, kernel_size=15, sigma=3.0, margin=10.0, mask_rate=0.7, use_logit_margin=True):
        super().__init__('TGR_Innovative_MDITI', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.num_diverse = num_diverse
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.margin = margin
        self.use_logit_margin = use_logit_margin
        self.mask_rate = mask_rate

        # Gaussian kernel for TI
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        self.ti_kernel = kernel_2d.expand(3, 1, kernel_size, kernel_size).cuda()

    def _create_diverse_inputs(self, x):
        diverse_inputs = [x]
        for _ in range(self.num_diverse - 1):
            transformed = x.clone()
            # 隨機resize+pad
            rnd = random.randint(224, 256)
            transformed = F.interpolate(transformed, size=(rnd, rnd), mode='bilinear', align_corners=False)
            pad_size = 256 - rnd
            if pad_size > 0:
                left = random.randint(0, pad_size)
                top = random.randint(0, pad_size)
                right = pad_size - left
                bottom = pad_size - top
                transformed = F.pad(transformed, (left, right, top, bottom), value=0)
            
            # 最後加這一行，確保尺寸正確
            transformed = F.interpolate(transformed, size=(224, 224), mode='bilinear', align_corners=False)
            diverse_inputs.append(transformed)
        return diverse_inputs

    def _apply_ti(self, grad):
        # TI: 平移不變性，對梯度做高斯模糊
        padding = self.kernel_size // 2
        return F.conv2d(grad, self.ti_kernel, padding=padding, groups=3)

    def _generate_random_mask(self, shape, rate):
        # 隨機遮罩，僅攻擊部分像素
        mask = torch.ones(shape).cuda()
        rand = torch.rand(shape).cuda()
        mask[rand > rate] = 0
        return mask

    def _compute_logit_margin_loss(self, outputs, labels):
        if not self.use_logit_margin:
            return F.cross_entropy(outputs, labels)
        correct_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        outputs_copy = outputs.clone()
        outputs_copy.scatter_(1, labels.unsqueeze(1), -float('inf'))
        max_wrong_logits = outputs_copy.max(1)[0]
        if self.target:
            loss = torch.clamp(correct_logits - max_wrong_logits + self.margin, min=0)
        else:
            loss = torch.clamp(max_wrong_logits - correct_logits + self.margin, min=0)
        return loss.mean()

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()
        for i in range(self.steps):
            current_adv = unnorm_inps + perts
            diverse_inputs = self._create_diverse_inputs(current_adv)
            total_grad = torch.zeros_like(perts)
            for input_variant in diverse_inputs:
                outputs = self.model(self._sub_mean_div_std(input_variant))
                loss = self.loss_flag * self._compute_logit_margin_loss(outputs, labels)
                loss.backward(retain_graph=True)
                if perts.grad is not None:
                    total_grad += perts.grad.data / len(diverse_inputs)
                    perts.grad.data.zero_()
            # 隨機遮罩
            mask = self._generate_random_mask(perts.shape, self.mask_rate)
            total_grad = total_grad * mask
            # TI: 對梯度做高斯模糊
            total_grad = self._apply_ti(total_grad)
            # Momentum
            grad_norm = torch.norm(total_grad.view(total_grad.shape[0], -1), dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            total_grad = total_grad / (grad_norm + 1e-8)
            momentum = self.decay * momentum + total_grad
            effective_grad = momentum
            # 更新擾動
            perts.data = self._update_perts(perts.data, effective_grad, self.alpha)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.data = torch.clamp(perts.data, -self.epsilon, self.epsilon)
            if (i + 1) % 5 == 0:
                perts.requires_grad_()
        final_adv = unnorm_inps + perts.data
        return (self._sub_mean_div_std(final_adv)).detach(), None

TGR = TGR_Innovative_MDITI