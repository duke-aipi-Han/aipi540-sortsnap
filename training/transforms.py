from torchvision import transforms

from src.preprocess import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_inference_transform


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.25,
                hue=0.03,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_minimal_train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms():
    return get_inference_transform()


def get_stress_test_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ColorJitter(brightness=(0.35, 0.75), contrast=(0.65, 1.0)),
            transforms.RandomRotation(degrees=18),
            transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
