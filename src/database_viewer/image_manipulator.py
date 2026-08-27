from torchvision.transforms import v2
from PIL.Image import Image

def manipulate_image(image: Image) -> Image:
    t = v2.PILToTensor()(image).unsqueeze(0)
    B, C, H, W = t.shape

    ft = v2.Compose([
        v2.RandomResizedCrop(
            size=(H, W),
            scale=(0.8, 1),
            ratio=(3/4, 4/3)
        ),
        v2.RandomErasing(
            p=0.5,
            scale=(0.01, 0.05), # (1%, 5%)
            ratio=(3/4, 4/3)
        ),
        v2.RandomErasing(
            p=0.5,
            scale=(0.01, 0.05), # (1%, 5%)
            ratio=(3/4, 4/3)
        ),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.7, 1.3),
            saturation=(0.7, 1.3),
            hue=(-0.05, 0.05)
        )
    ])

    t = ft(t)

    return v2.ToPILImage()(t.squeeze(0))
