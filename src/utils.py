import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FewShotDataset(Dataset):
    def __init__(self, root_dir, n_way=3, k_shot=5, q_query=5, class_list=None, augment=False):
        self.root_dir = root_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform = train_transform if augment else eval_transform

        all_classes = class_list if class_list is not None else sorted(os.listdir(root_dir))

        # Build image list per class, filtering valid image files only
        self.class_images = {}
        self.classes = []
        for cls in all_classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = [f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) >= k_shot + q_query:
                self.class_images[cls] = imgs
                self.classes.append(cls)

        print(f"  Dataset ready: {len(self.classes)} classes")
        for cls in self.classes:
            print(f"    {cls}: {len(self.class_images[cls])} images")

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        if len(self.classes) < self.n_way:
            return [], [], [], []

        # Sample n_way classes — ensures all classes used equally
        selected_classes = random.sample(self.classes, self.n_way)
        support_images, query_images = [], []
        support_labels, query_labels = [], []

        valid = True
        for i, cls in enumerate(selected_classes):
            imgs = self.class_images[cls]
            if len(imgs) < self.k_shot + self.q_query:
                valid = False
                break
            selected = random.sample(imgs, self.k_shot + self.q_query)
            support_imgs = selected[:self.k_shot]
            query_imgs   = selected[self.k_shot:]

            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in support_imgs:
                img = self.transform(Image.open(os.path.join(cls_dir, img_name)).convert("RGB"))
                support_images.append(img)
                support_labels.append(i)

            for img_name in query_imgs:
                img = self.transform(Image.open(os.path.join(cls_dir, img_name)).convert("RGB"))
                query_images.append(img)
                query_labels.append(i)

        if not valid:
            return [], [], [], []

        return support_images, support_labels, query_images, query_labels