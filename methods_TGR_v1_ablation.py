import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
import scipy.stats as st
from functools import partial
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
                 use_adaptive_step=True, use_gradient_smoothing=True,
                 enable_attn_hook=False, enable_q_hook=True, enable_v_hook=True, enable_mlp_hook=True):
        super(TGR_v1, self).__init__('TGR_v1', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.base_step_size = self.epsilon / self.steps
        self.decay = decay

        # ablation hook switches
        self.enable_attn_hook = enable_attn_hook
        self.enable_q_hook = enable_q_hook
        self.enable_v_hook = enable_v_hook
        self.enable_mlp_hook = enable_mlp_hook

        # attack enhancements
        self.use_diverse_inputs = use_diverse_inputs
        self.use_translation_invariant = use_translation_invariant
        self.use_scale_invariant = use_scale_invariant
        self.use_momentum_variance = use_momentum_variance
        self.use_logit_margin = use_logit_margin
        self.use_adaptive_step = use_adaptive_step
        self.use_gradient_smoothing = use_gradient_smoothing
        
        self.num_copies = 5
        self.prob_transform = 0.5

        self.scale_copies = 3
        self.scale_range = [0.9, 1.1]
        self.margin = 20.0
        self.smooth_kernel_size = 3
        self.step_increase_factor = 1.1
        self.step_decrease_factor = 0.9
        self.success_threshold = 0.1

        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches
        
        if self.use_gradient_smoothing:
            self._init_smooth_kernel()
        
        self._register_model()

    def _init_smooth_kernel(self):
        kernel_size = self.smooth_kernel_size
        sigma = 1.0
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        self.smooth_kernel = kernel_2d.expand(3, 1, kernel_size, kernel_size).cuda()

    def _apply_gradient_smoothing(self, grad):
        if not self.use_gradient_smoothing:
            return grad
        padding = self.smooth_kernel_size // 2
        smoothed_grad = F.conv2d(grad, self.smooth_kernel, padding=padding, groups=3)
        return smoothed_grad

    def _create_diverse_inputs(self, x):
        if not self.use_diverse_inputs:
            return [x]
        diverse_inputs = [x]
        for _ in range(self.num_copies - 1):
            transformed = x.clone()
            if torch.rand(1) < self.prob_transform:
                pad_size = random.randint(0, 30)
                if pad_size > 0:
                    transformed = F.pad(transformed, [pad_size] * 4, mode='constant', value=0)
                    crop_size = transformed.shape[-1] - pad_size
                    start = random.randint(0, pad_size)
                    transformed = transformed[:, :, start:start+crop_size, start:start+crop_size]
                    transformed = F.interpolate(transformed, size=(224, 224), mode='bilinear', align_corners=False)
            if torch.rand(1) < self.prob_transform:
                if torch.rand(1) < 0.5:
                    transformed = torch.flip(transformed, dims=[3])
            if torch.rand(1) < self.prob_transform:
                angle = random.uniform(-5, 5)
                if abs(angle) > 0.1:
                    shift_x = int(transformed.shape[-1] * angle / 45)
                    shift_y = int(transformed.shape[-2] * angle / 45)
                    if abs(shift_x) > 0 or abs(shift_y) > 0:
                        transformed = torch.roll(transformed, shifts=(shift_y, shift_x), dims=(-2, -1))
            diverse_inputs.append(transformed)
        return diverse_inputs

    def _create_scale_invariant_inputs(self, x):
        if not self.use_scale_invariant:
            return [x]
        scale_inputs = []
        for _ in range(self.scale_copies):
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            new_size = int(224 * scale)
            scaled = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=False)
            if new_size > 224:
                start = (new_size - 224) // 2
                scaled = scaled[:, :, start:start+224, start:start+224]
            elif new_size < 224:
                pad_size = (224 - new_size) // 2
                scaled = F.pad(scaled, [pad_size, 224-new_size-pad_size, pad_size, 224-new_size-pad_size])
            scale_inputs.append(scaled)
        return scale_inputs

    def _apply_translation_invariant(self, x):
        if not self.use_translation_invariant:
            return x
        shift_x = random.randint(-10, 10)
        shift_y = random.randint(-10, 10)
        if shift_x != 0 or shift_y != 0:
            x = torch.roll(x, shifts=(shift_y, shift_x), dims=(-2, -1))
        return x

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

    def _adaptive_step_size(self, step, success_rate):
        if not self.use_adaptive_step:
            return self.base_step_size
        factor = self.step_decrease_factor if success_rate > self.success_threshold else self.step_increase_factor
        decay_factor = 1.0 - (step / self.steps) * 0.3
        adaptive_step = self.base_step_size * factor * decay_factor
        return max(adaptive_step, self.base_step_size * 0.5)

    def _register_model(self):
        # Q hook
        def q_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            res = [out_grad]
            for idx in range(1, len(grad_in)):
                res.append(grad_in[idx])
            return tuple(res)
        # V hook
        def v_tgr(module, grad_in, grad_out, gamma):
            if grad_in[0] is None:
                return
            res = [grad_in[0]]
            for idx in range(1, len(grad_in)):
                res.append(grad_in[idx])
            return tuple(res)
        # Attention hook
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
            return (out_grad, )
        # MLP hook
        def mlp_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if len(out_grad.shape) == 2:
                B, C = out_grad.shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu, axis=0)
                min_all = np.argmin(out_grad_cpu, axis=0)
                out_grad[max_all, range(C)] = 0.0
                out_grad[min_all, range(C)] = 0.0
            elif len(out_grad.shape) == 4:
                B, C, H, W = out_grad.shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                for c in range(C):
                    h = int(np.clip(max_all_H[c], 0, H-1))
                    w = int(np.clip(max_all_W[c], 0, W-1))
                    out_grad[:, c, h, w] = 0.0
            return (out_grad,)

        attn_tgr_hook = partial(attn_tgr, gamma=0.5)
        q_tgr_hook = partial(q_tgr, gamma=0.9)
        v_tgr_hook = partial(v_tgr, gamma=0.9)
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.7)

        # Register hooks according to ablation switches
        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                if self.enable_attn_hook:
                    self.model.blocks[i].attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                if self.enable_q_hook and hasattr(self.model.blocks[i].attn, 'q'):
                    self.model.blocks[i].attn.q.register_full_backward_hook(q_tgr_hook)
                if self.enable_v_hook and hasattr(self.model.blocks[i].attn, 'v'):
                    self.model.blocks[i].attn.v.register_full_backward_hook(v_tgr_hook)
                if self.enable_mlp_hook:
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
                blk = self.model.transformers[transformer_ind].blocks[used_block_ind]
                if self.enable_attn_hook:
                    blk.attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                if self.enable_q_hook and hasattr(blk.attn, 'q'):
                    blk.attn.q.register_full_backward_hook(q_tgr_hook)
                if self.enable_v_hook and hasattr(blk.attn, 'v'):
                    blk.attn.v.register_full_backward_hook(v_tgr_hook)
                if self.enable_mlp_hook:
                    blk.mlp.register_full_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    blk = self.model.stage2[block_ind]
                else:
                    blk = self.model.stage3[block_ind-4]
                if self.enable_attn_hook:
                    blk.attn.attn_drop.register_full_backward_hook(attn_tgr_hook)
                if self.enable_q_hook and hasattr(blk.attn, 'q'):
                    blk.attn.q.register_full_backward_hook(q_tgr_hook)
                if self.enable_v_hook and hasattr(blk.attn, 'v'):
                    blk.attn.v.register_full_backward_hook(v_tgr_hook)
                if self.enable_mlp_hook:
                    blk.mlp.register_full_backward_hook(mlp_tgr_hook)
        # Add elif for cait/other models as needed

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        ce_loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(inps).cuda()
        variance = torch.zeros_like(inps).cuda() if self.use_momentum_variance else None
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()
        success_history = []

        for i in range(self.steps):
            current_adv = unnorm_inps + perts
            diverse_inputs = self._create_diverse_inputs(current_adv)
            scale_inputs = self._create_scale_invariant_inputs(current_adv)
            all_inputs = diverse_inputs + scale_inputs
            total_grad = torch.zeros_like(perts)
            for input_variant in all_inputs:
                input_variant = self._apply_translation_invariant(input_variant)
                outputs = self.model(self._sub_mean_div_std(input_variant))
                if self.use_logit_margin:
                    loss = self.loss_flag * self._compute_logit_margin_loss(outputs, labels)
                else:
                    loss = self.loss_flag * ce_loss(outputs, labels)
                loss.backward(retain_graph=True)
                if perts.grad is not None:
                    total_grad += perts.grad.data / len(all_inputs)
                    perts.grad.data.zero_()
            total_grad = self._apply_gradient_smoothing(total_grad)
            grad_norm = torch.norm(total_grad.view(total_grad.shape[0], -1), dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            total_grad = total_grad / (grad_norm + 1e-8)
            momentum = self.decay * momentum + total_grad
            if self.use_momentum_variance:
                variance = self.decay * variance + (1 - self.decay) * (total_grad ** 2)
                effective_grad = momentum / (torch.sqrt(variance) + 1e-8)
            else:
                effective_grad = momentum
            with torch.no_grad():
                current_outputs = self.model(self._sub_mean_div_std(current_adv))
                pred_labels = current_outputs.argmax(dim=1)
                success_rate = float((pred_labels != labels).sum()) / len(labels)
                success_history.append(success_rate)
            current_step_size = self._adaptive_step_size(i, success_rate)
            perts.data = self._update_perts(perts.data, effective_grad, current_step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            if (i + 1) % 5 == 0:
                perts.requires_grad_()
        final_adv = unnorm_inps + perts.data
        return (self._sub_mean_div_std(final_adv)).detach(), None

TGR = TGR_v1