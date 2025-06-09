# TGR_v1
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
        if self.target:
            self.loss_flag = -1
        else:
            self.loss_flag = 1
        self.used_params = params(self.model_name)

        # loading model
        self.model = get_model(self.model_name)
        self.model.cuda()
        self.model.eval()

    def forward(self, *input):
        """
        Rewrite
        """
        raise NotImplementedError

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

    def _update_inps(self, inps, grad, step_size):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        unnorm_inps = unnorm_inps + step_size * grad.sign()
        unnorm_inps = torch.clamp(unnorm_inps, min=0, max=1).detach()
        adv_inps = self._sub_mean_div_std(unnorm_inps)
        return adv_inps

    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def _return_perts(self, clean_inps, inps):
        clean_unnorm = self._mul_std_add_mean(clean_inps.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inps.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images


class TGR_v1(BaseAttack):
    def __init__(self, model_name, sample_num_batches=130, steps=10, epsilon=16/255, target=False, 
                 decay=1.0, use_diverse_inputs=True, use_translation_invariant=True, 
                 use_scale_invariant=True, use_momentum_variance=True, use_logit_margin=True,
                 use_adaptive_step=True, use_gradient_smoothing=True):
        super(TGR_v1, self).__init__('TGR_Enhanced', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.base_step_size = self.epsilon / self.steps
        self.decay = decay

        # 攻擊效果增強參數
        self.use_diverse_inputs = use_diverse_inputs
        self.use_translation_invariant = use_translation_invariant
        self.use_scale_invariant = use_scale_invariant
        self.use_momentum_variance = use_momentum_variance
        self.use_logit_margin = use_logit_margin
        self.use_adaptive_step = use_adaptive_step
        self.use_gradient_smoothing = use_gradient_smoothing
        
        # 多樣化輸入參數
        self.num_copies = 5  # 每次迭代使用的副本數量
        self.prob_transform = 0.5  # 應用變換的概率
        
        # 縮放不變性參數
        self.scale_copies = 3
        self.scale_range = [0.9, 1.1]
        
        # Logit margin 攻擊參數
        self.margin = 20.0  # logit margin
        
        # 梯度平滑參數
        self.smooth_kernel_size = 3
        
        # 自適應步長參數
        self.step_increase_factor = 1.1
        self.step_decrease_factor = 0.9
        self.success_threshold = 0.1  # 成功的logit差異閾值

        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches
        
        # 初始化梯度平滑核
        if self.use_gradient_smoothing:
            self._init_smooth_kernel()
        
        self._register_model()

    def _init_smooth_kernel(self):
        """初始化用於梯度平滑的高斯核"""
        kernel_size = self.smooth_kernel_size
        sigma = 1.0
        
        # 創建1D高斯核
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # 擴展為2D核
        kernel_2d = gauss[:, None] * gauss[None, :]
        self.smooth_kernel = kernel_2d.expand(3, 1, kernel_size, kernel_size).cuda()

    def _apply_gradient_smoothing(self, grad):
        """應用梯度平滑"""
        if not self.use_gradient_smoothing:
            return grad
        
        padding = self.smooth_kernel_size // 2
        smoothed_grad = F.conv2d(grad, self.smooth_kernel, padding=padding, groups=3)
        return smoothed_grad

    def _create_diverse_inputs(self, x):
        """創建多樣化的輸入副本"""
        if not self.use_diverse_inputs:
            return [x]
        
        diverse_inputs = [x]
        
        for _ in range(self.num_copies - 1):
            transformed = x.clone()
            
            # 隨機變換
            if torch.rand(1) < self.prob_transform:
                # 隨機填充和裁剪
                pad_size = random.randint(0, 30)
                if pad_size > 0:
                    transformed = F.pad(transformed, [pad_size] * 4, mode='constant', value=0)
                    crop_size = transformed.shape[-1] - pad_size
                    start = random.randint(0, pad_size)
                    transformed = transformed[:, :, start:start+crop_size, start:start+crop_size]
                    transformed = F.interpolate(transformed, size=(224, 224), mode='bilinear', align_corners=False)
            
            if torch.rand(1) < self.prob_transform:
                # 隨機翻轉
                if torch.rand(1) < 0.5:
                    transformed = torch.flip(transformed, dims=[3])  # 水平翻轉
            
            if torch.rand(1) < self.prob_transform:
                # 輕微旋轉 (通過裁剪和填充近似)
                angle = random.uniform(-5, 5)  # 小角度旋轉
                if abs(angle) > 0.1:
                    # 簡單的平移變換來模擬小角度旋轉
                    shift_x = int(transformed.shape[-1] * angle / 45)  # 近似變換
                    shift_y = int(transformed.shape[-2] * angle / 45)
                    if abs(shift_x) > 0 or abs(shift_y) > 0:
                        transformed = torch.roll(transformed, shifts=(shift_y, shift_x), dims=(-2, -1))
            
            diverse_inputs.append(transformed)
        
        return diverse_inputs

    def _create_scale_invariant_inputs(self, x):
        """創建縮放不變的輸入"""
        if not self.use_scale_invariant:
            return [x]
        
        scale_inputs = []
        for i in range(self.scale_copies):
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            new_size = int(224 * scale)
            
            # 調整大小
            scaled = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=False)
            
            # 居中裁剪或填充到原始大小
            if new_size > 224:
                # 裁剪
                start = (new_size - 224) // 2
                scaled = scaled[:, :, start:start+224, start:start+224]
            elif new_size < 224:
                # 填充
                pad_size = (224 - new_size) // 2
                scaled = F.pad(scaled, [pad_size, 224-new_size-pad_size, pad_size, 224-new_size-pad_size])
            
            scale_inputs.append(scaled)
        
        return scale_inputs

    def _apply_translation_invariant(self, x):
        """應用平移不變性"""
        if not self.use_translation_invariant:
            return x
        
        # 隨機平移
        shift_x = random.randint(-10, 10)
        shift_y = random.randint(-10, 10)
        
        if shift_x != 0 or shift_y != 0:
            x = torch.roll(x, shifts=(shift_y, shift_x), dims=(-2, -1))
        
        return x

    def _compute_logit_margin_loss(self, outputs, labels):
        """計算 logit margin 損失"""
        if not self.use_logit_margin:
            return F.cross_entropy(outputs, labels)
        
        # 獲取正確類別的logit
        correct_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # 獲取最大的錯誤類別logit
        outputs_copy = outputs.clone()
        outputs_copy.scatter_(1, labels.unsqueeze(1), -float('inf'))
        max_wrong_logits = outputs_copy.max(1)[0]
        
        # Margin loss - 增大正確類別和錯誤類別之間的差距
        if self.target:
            # 目標攻擊：減少目標類別的logit
            loss = torch.clamp(correct_logits - max_wrong_logits + self.margin, min=0)
        else:
            # 非目標攻擊：增加錯誤類別的logit
            loss = torch.clamp(max_wrong_logits - correct_logits + self.margin, min=0)
        
        return loss.mean()

    def _adaptive_step_size(self, step, success_rate):
        """自適應調整步長"""
        if not self.use_adaptive_step:
            return self.base_step_size
        
        # 根據成功率調整步長
        if success_rate > self.success_threshold:
            # 攻擊成功率高，減小步長保持穩定性
            factor = self.step_decrease_factor
        else:
            # 攻擊成功率低，增大步長增強攻擊力
            factor = self.step_increase_factor
        
        # 步長隨迭代逐漸衰減
        decay_factor = 1.0 - (step / self.steps) * 0.3
        adaptive_step = self.base_step_size * factor * decay_factor
        
        return max(adaptive_step, self.base_step_size * 0.5)  # 最小步長限制

    def _register_model(self):   
        def attn_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            gshape = out_grad.shape
            if len(gshape) == 4:
                B, C, H, W = gshape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = np.clip(max_all // H, 0, H-1)
                max_all_W = np.clip(max_all % H, 0, W-1)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = np.clip(min_all // H, 0, H-1)
                min_all_W = np.clip(min_all % H, 0, W-1)
                idx_C = np.arange(C)
                for c in idx_C:
                    out_grad[:, c, max_all_H[c], :] = 0.0
                    out_grad[:, c, :, max_all_W[c]] = 0.0
                    out_grad[:, c, min_all_H[c], :] = 0.0
                    out_grad[:, c, :, min_all_W[c]] = 0.0
            elif len(gshape) == 4 and gshape[1] != gshape[3]:  # CAIT: [B, H, W, C]
                B, H, W, C = gshape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, H*W, C)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                max_all_H = np.clip(max_all // H, 0, H-1)
                max_all_W = np.clip(max_all % H, 0, W-1)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                min_all_H = np.clip(min_all // H, 0, H-1)
                min_all_W = np.clip(min_all % H, 0, W-1)
                idx_C = np.arange(C)
                for c in idx_C:
                    out_grad[:, max_all_H[c], :, c] = 0.0
                    out_grad[:, :, max_all_W[c], c] = 0.0
                    out_grad[:, min_all_H[c], :, c] = 0.0
                    out_grad[:, :, min_all_W[c], c] = 0.0
            return (out_grad, )
        def attn_cait_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            
            B,H,W,C = grad_in[0].shape
            out_grad_cpu = out_grad.data.clone().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0,:,0,:], axis = 0)
            min_all = np.argmin(out_grad_cpu[0,:,0,:], axis = 0)
                
            out_grad[:,max_all,:,range(C)] = 0.0
            out_grad[:,min_all,:,range(C)] = 0.0
            return (out_grad, )
            
        def q_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            # 回傳的 tuple 長度必須和 grad_in 一致
            res = [out_grad]
            for idx in range(1, len(grad_in)):
                res.append(grad_in[idx])
            return tuple(res)
            
        def v_tgr(module, grad_in, grad_out, gamma):
            if grad_in[0] is None:
                return
            # 簡化處理，直接返回梯度
            return
                
        def mlp_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if self.model_name == 'visformer_small':
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                out_grad[:,range(C),max_all_H,max_all_W] = 0.0

            elif self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                if len(out_grad.shape) == 2:
                    B, C = out_grad.shape
                    out_grad_cpu = out_grad.data.clone().cpu().numpy()
                    max_all = np.argmax(out_grad_cpu, axis=0)
                    min_all = np.argmin(out_grad_cpu, axis=0)
                    out_grad[max_all, range(C)] = 0.0
                    out_grad[min_all, range(C)] = 0.0

            return (out_grad,)

        # 增強的hook，提高gamma值以增強攻擊效果
        attn_tgr_hook = partial(attn_tgr, gamma=0.5)  # 從0.25提高到0.5
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=0.5)
        v_tgr_hook = partial(v_tgr, gamma=0.9)  # 從0.75提高到0.9
        q_tgr_hook = partial(q_tgr, gamma=0.9)
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.7)  # 從0.5提高到0.7

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                self.model.blocks[i].attn.qkv.register_full_backward_hook(v_tgr_hook)
                self.model.blocks[i].mlp.register_full_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_full_backward_hook(v_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_full_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                    self.model.blocks[block_ind].attn.qkv.register_full_backward_hook(v_tgr_hook)
                    self.model.blocks[block_ind].mlp.register_full_backward_hook(mlp_tgr_hook)
                elif block_ind >= 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_full_backward_hook(attn_cait_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.q.register_full_backward_hook(q_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.k.register_full_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.v.register_full_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].mlp.register_full_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                    self.model.stage2[block_ind].attn.qkv.register_full_backward_hook(v_tgr_hook)
                    self.model.stage2[block_ind].mlp.register_full_backward_hook(mlp_tgr_hook)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                    self.model.stage3[block_ind-4].attn.qkv.register_full_backward_hook(v_tgr_hook)
                    self.model.stage3[block_ind-4].mlp.register_full_backward_hook(mlp_tgr_hook)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        
        # 多種損失函數組合
        ce_loss = nn.CrossEntropyLoss()
        
        # 初始化動量
        momentum = torch.zeros_like(inps).cuda()
        variance = torch.zeros_like(inps).cuda() if self.use_momentum_variance else None
        
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()
        
        success_history = []

        for i in range(self.steps):
            current_adv = unnorm_inps + perts
            
            # 創建多樣化輸入
            diverse_inputs = self._create_diverse_inputs(current_adv)
            scale_inputs = self._create_scale_invariant_inputs(current_adv)
            all_inputs = diverse_inputs + scale_inputs
            
            # 計算所有輸入的梯度
            total_grad = torch.zeros_like(perts)
            total_loss = 0
            
            for input_variant in all_inputs:
                # 應用平移不變性
                input_variant = self._apply_translation_invariant(input_variant)
                
                # 前向傳播
                outputs = self.model(self._sub_mean_div_std(input_variant))
                
                # 計算損失
                if self.use_logit_margin:
                    loss = self.loss_flag * self._compute_logit_margin_loss(outputs, labels)
                else:
                    loss = self.loss_flag * ce_loss(outputs, labels)
                
                total_loss += loss
                
                # 計算梯度
                loss.backward(retain_graph=True)
                if perts.grad is not None:
                    total_grad += perts.grad.data / len(all_inputs)
                    perts.grad.data.zero_()
            
            # 應用梯度平滑
            total_grad = self._apply_gradient_smoothing(total_grad)
            
            # 梯度正規化 - 使用更強的正規化
            grad_norm = torch.norm(total_grad.view(total_grad.shape[0], -1), dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            total_grad = total_grad / (grad_norm + 1e-8)
            
            # 動量更新
            momentum = self.decay * momentum + total_grad
            
            # 方差動量 (Momentum Variance)
            if self.use_momentum_variance:
                variance = self.decay * variance + (1 - self.decay) * (total_grad ** 2)
                effective_grad = momentum / (torch.sqrt(variance) + 1e-8)
            else:
                effective_grad = momentum
            
            # 計算成功率用於自適應步長
            with torch.no_grad():
                current_outputs = self.model(self._sub_mean_div_std(current_adv))
                pred_labels = current_outputs.argmax(dim=1)
                success_rate = float((pred_labels != labels).sum()) / len(labels)
                success_history.append(success_rate)
            
            # 自適應步長
            current_step_size = self._adaptive_step_size(i, success_rate)
            
            # 更新擾動
            perts.data = self._update_perts(perts.data, effective_grad, current_step_size)
            
            # 確保擾動在有效範圍內
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            
            # 每隔幾步重新計算梯度，避免梯度消失
            if (i + 1) % 5 == 0:
                perts.requires_grad_()
        
        final_adv = unnorm_inps + perts.data
        return (self._sub_mean_div_std(final_adv)).detach(), None


# 為了向後兼容，保留原來的類別名稱作為別名
TGR = TGR_v1