import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoConfig
from transformers import SiglipImageProcessor, SiglipVisionModel

class SigLIP2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            #model, _ = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower= SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # len 25, each is (batch, 577, 1024) - (batch, 729, 1152)
        if self.select_feature == 'patch':
            image_features = image_features[:, :] # not remove the first token --> (batch, 576, 1024) - (batch, 729, 1152)
        elif self.select_feature == 'cls_patch': 
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images): # images tensor
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.config.vision_config.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype # torch.float16

    @property
    def device(self):
        return self.vision_tower.device # device(type='cuda', index=0)

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config # CLIPVisionConfig - SiglipConfig
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.vision_config.hidden_size # 1152

    @property
    def num_patches_per_side(self):
        return self.config.vision_config.image_size // self.config.vision_config.patch_size # (348 // 14) = 27

    @property
    def num_patches(self):
        return (self.config.vision_config.image_size // self.config.vision_config.patch_size) ** 2 # 729
