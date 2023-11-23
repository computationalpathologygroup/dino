import random

from torchvision import transforms
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PatchDataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        global_crop_size = 224
        local_crop_size = 96

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crop_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, x):
        crops = []
        crops.append(self.global_transfo1(x))
        crops.append(self.global_transfo2(x))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x))
        return crops


class RegionDataAugmentationDINO(object):
    """
    Modified Data Augmentaton for DINO for [region_size x region_size] resolutions for performing local / global crops on features in image grid
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_number,
        local_crops_scale,
        region_size: int = 4096,
        patch_size: int = 256,
    ):
        self.npatch = int(region_size // patch_size)
        global_crop_size = int(global_crops_scale * self.npatch)
        local_crop_size = int(local_crops_scale * self.npatch)

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomCrop(global_crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomCrop(global_crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomCrop(local_crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __call__(self, x):
        crops = []
        x = x.unfold(0, self.npatch, self.npatch).transpose(
            0, 1
        )  # [m, 384] -> [npatch, 384, npatch] -> [384, npatch, npatch]
        crops.append(self.global_transfo1(x))
        crops.append(self.global_transfo2(x))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x))
        return crops