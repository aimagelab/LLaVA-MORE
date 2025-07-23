import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .siglip_encoder import SigLIPVisionTower, SigLIPVisionTowerS2
from .dino_encoder import DINOVisionTower
from .dino_with_register_encoder import DINORegisterVisionTower
from .vit_encoder import ViTVisionTower
from .i_jepa_encoder import IJepaVisionTower
from .siglip_2_encoder import SigLIP2VisionTower

# separate the N different visual backbones in different file (to improve readability)
# in each file a single backbone and in case the associated implementation of S2

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    use_siglip = getattr(vision_tower_cfg, 'siglip', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("timm") or vision_tower.startswith("google") or "ShareGPT4V" in vision_tower or 'facebook' in vision_tower:
        if 'ijepa' in vision_tower:
            return IJepaVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if 'google--vit-large' in vision_tower:
            return ViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if 'dino' in vision_tower and 'registers' in vision_tower:
            return DINORegisterVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if 'dino' in vision_tower and 'registers' not in vision_tower:
            return DINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if use_s2 and use_siglip:
            return SigLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        if use_siglip and 'siglip2' in vision_tower:
            return SigLIP2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        if use_siglip:
            return SigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
