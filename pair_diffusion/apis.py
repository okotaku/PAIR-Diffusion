import einops
import numpy as np
import torch

from huggingface_hub import hf_hub_url, hf_hub_download

from pair_diffusion.cldm.model import create_model, load_state_dict
from pair_diffusion.cldm.ddim_hacked import DDIMSamplerSpaCFG
from pair_diffusion.ldm.models.autoencoder import DiagonalGaussianDistribution

urls = {
    'PAIR/PAIR-diffusion-sdv15-coco-finetune': ['pair_diffusion_epoch62.ckpt']
}

WTS_DICT = {

}

for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)

        WTS_DICT[repo] = hf_hub_download(repo_id=repo, filename=file)

class PairDiffusion:
    def __init__(self):
        self.kernel = np.ones((5, 5), np.uint8)
        self.num_samples = 1
        self.ddim_steps = 20
        self.strength = 1.0
        scale_s = 5.
        scale_f = 8.
        scale_t = 9.
        self.scale = [scale_s, scale_f, scale_t]
        self.eta = 0.
        self.inter = 1.
        self.save_memory = False
        
        self.model = create_model('./configs/sap_fixed_hintnet_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict(WTS_DICT['PAIR/PAIR-diffusion-sdv15-coco-finetune'], location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSamplerSpaCFG(self.model)

    def edit(self, input_img, ref_img, input_mask, ref_mask, H, W):
        _, mean_feat_inpt, one_hot_inpt, _ = self.model.get_appearance(input_img, input_mask, return_all=True)

        _, mean_feat_ref, _, _ = self.model.get_appearance(ref_img, ref_mask, return_all=True)

        assert mean_feat_ref.shape[1] > 1
        mean_feat_inpt[:, -1] = (1 - self.inter) * mean_feat_inpt[:, -1] + self.inter * mean_feat_ref[:, 1]

        splatted_feat = torch.einsum('nmc, nmhw->nchw', mean_feat_inpt, one_hot_inpt)
        appearance = torch.nn.functional.normalize(splatted_feat) #l2 normaliz
        structure = torch.nn.functional.interpolate(input_mask, (H, W))
        appearance = torch.nn.functional.interpolate(appearance, (H, W))

        return structure, appearance
    
    def process(self, input_img, ref_img, input_mask, ref_mask, prompt,
                a_prompt='best quality, extremely detailed',
                n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
                ):
        # preprocess
        H, W = input_img.shape[:2]
        input_img = (input_img / 127.5 - 1)
        input_img =  torch.from_numpy(input_img.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2)

        ref_img = (ref_img / 127.5 - 1)
        ref_img =  torch.from_numpy(ref_img.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2)

        input_mask = torch.from_numpy(input_mask.astype(np.float32)).float().cuda().unsqueeze(0).unsqueeze(1)
        ref_mask = torch.from_numpy(ref_mask.astype(np.float32)).float().cuda().unsqueeze(0).unsqueeze(1)
        
        structure, appearance = self.edit(input_img, ref_img, input_mask, ref_mask, H, W)

        null_structure = torch.zeros(structure.shape).cuda() - 1
        null_appearance = torch.zeros(appearance.shape).cuda()

        null_control = torch.cat([null_structure, null_appearance], dim=1)
        structure_control = torch.cat([structure, null_appearance], dim=1)
        full_control = torch.cat([structure, appearance], dim=1)

        null_control = torch.cat([null_control for _ in range(self.num_samples)], dim=0)
        structure_control = torch.cat([structure_control for _ in range(self.num_samples)], dim=0)
        full_control = torch.cat([full_control for _ in range(self.num_samples)], dim=0)

        x0 = self.model.encode_first_stage(input_img)
        x0 = x0.sample() if isinstance(x0, DiagonalGaussianDistribution) else x0 # todo: check if we can set random number
        x0 = x0 * self.model.scale_factor
        input_mask = 1 - torch.tensor(input_mask).unsqueeze(0).unsqueeze(1).cuda()
        input_mask = torch.nn.functional.interpolate(input_mask[0, 0], x0.shape[2:]).float()

        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        uc_cross = self.model.get_learned_conditioning([n_prompt] * self.num_samples)
        cond = {"c_concat": [full_control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt]  * self.num_samples)]}
        un_cond = {"c_concat": [null_control], "c_crossattn": [uc_cross]}
        un_cond_struct = {"c_concat": [structure_control], "c_crossattn": [uc_cross]}
        un_cond_struct_app = {"c_concat": [full_control], "c_crossattn": [uc_cross]}

        shape = (4, H // 8, W // 8)

        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        self.model.control_scales = ([self.strength] * 13)
        samples, _ = self.ddim_sampler.sample(self.ddim_steps, self.num_samples,
                                        shape, cond, verbose=False, eta=self.eta,
                                        unconditional_guidance_scale=self.scale, mask=input_mask, x0=x0,
                                        unconditional_conditioning=[un_cond, un_cond_struct, un_cond_struct_app ])

        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = (self.model.decode_first_stage(samples) + 1) * 127.5
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(self.num_samples)]
        return results
