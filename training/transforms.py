from torchvision import transforms

from src.preprocess import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_inference_transform

# apply some data augmentation to the training data to help it generalize better in "real world" scenarios

def get_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=20,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                shear=8,
            ),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.3),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.3,
                        hue=0.04,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomGrayscale(p=0.08),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.8))],
                p=0.35,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
            ),
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
