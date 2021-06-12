"""Office and Caltech10"""
from __future__ import print_function

import os
from torchvision.datasets.folder import default_loader, ImageFolder

from fade.file import FileManager

ALL_SOURCES = ["office31", "officehome65"]
ALL_DOMAINS = {
    "office31": ["amazon", "dslr", "webcam"],
    "officehome65": ["Art", "Clipart", "Product", "RealWorld"],
}
ALL_URLS = {
    # links from https://github.com/tim-learn/SHOT
    "office31": "https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view",
    "officehome65": "https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw"
}


class LoadImageFolder(ImageFolder):
    """Different from ImageFolder, you need to use transform to load images. If transform is None,
    then the default loader is used to load images. Otherwise, transform has to process the path str
    to load images.
    """

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.transform is None:
            sample = self.loader(path)
        else:
            sample = self.transform(path)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DefaultImageLoader(object):
    """Transformer to load image from path."""
    def __init__(self):
        pass

    def __call__(self, path):
        return default_loader(path)


def get_office_caltech_dataset(source, domain, transform=None, target_transform=None,
                               feature_type="images", load_img_by_transform=False):
    """load_img_by_transform: The dataset will not auto load image for transform.
    Use transform to process path str and transform path to images, instead."""
    root = FileManager.data(os.path.join(source), is_dir=True)
    if source.lower() == "office31":
        assert feature_type == "images", "Office31 only support image features."
        assert domain in ALL_DOMAINS[source.lower()], f"Unknown domain: {domain}"
        image_path = os.path.join(root, domain, "images")
        if not os.path.exists(image_path):
            print(f"### cwd: {os.getcwd()}")
            raise FileNotFoundError(f"No found image directory at: {image_path}. "
                                    f"Download zip file from {ALL_URLS[source.lower()]} and unpack"
                                    f" into {root}. Verify the file structure to make"
                                    f" sure the missing image path exist.")
        if load_img_by_transform:
            ds = LoadImageFolder(root=image_path, transform=transform,
                                 target_transform=target_transform)
        else:
            ds = ImageFolder(root=image_path, transform=transform,
                             target_transform=target_transform)
    elif source.lower() == "officehome65":
        assert feature_type == "images", "OfficeHome65 only support image features."
        assert domain in ALL_DOMAINS[source.lower()], f"Unknown domain: {domain}"
        image_path = os.path.join(root, domain)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No found image directory at: {image_path}. "
                                    f"Download zip file from {ALL_URLS[source.lower()]} and unpack"
                                    f" into {root}. Verify the file structure to make"
                                    f" sure the missing image path exist.")
        if load_img_by_transform:
            ds = LoadImageFolder(root=image_path, transform=transform,
                                 target_transform=target_transform)
        else:
            ds = ImageFolder(root=image_path, transform=transform,
                             target_transform=target_transform)
    else:
        raise ValueError(f"Invalid source: {source}")
    return ds


def main():
    """Verify the consistence of classes."""
    for source in ALL_SOURCES:
        class_to_idx = []
        for domain in ALL_DOMAINS[source]:
            print()
            print(f"====== source: {source}, domain: {domain} ======")
            ds = get_office_caltech_dataset(source, domain)
            print(f"  classes: {ds.classes}")
            print(f"  class_to_idx: {ds.class_to_idx}")
            if len(class_to_idx) == 0:
                pass
            else:
                for (name0, idx0), (name1, idx1) in zip(class_to_idx[-1].items(), ds.class_to_idx.items()):
                    assert name0 == name1
                    assert idx0 == idx1
                print(f"[OK] {len(ds.classes)} classes in domain {domain} of {source} are verified.")
            class_to_idx.append(ds.class_to_idx)


if __name__ == '__main__':
    main()
