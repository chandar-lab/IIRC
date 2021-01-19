# from torchvision import datasets
import argparse
import requests
import os
import hashlib
import warnings
import tarfile
import logging

def _download_cifar100(root="data"):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    os.makedirs(root, exist_ok=True)
    target_path = os.path.join(root, 'cifar-100-python.tar.gz')
    if os.path.exists(target_path):
        warnings.warn(f"{target_path} already exists, the dataset won't be download")
        return None
    print("downloading CIFAR 100")
    logging.info("downloading dataset")
    response = requests.get(url, stream=True)
    assert response.status_code == 200
    with open(target_path, 'wb') as f:
        f_content = response.raw.read()
        assert hashlib.md5(f_content).hexdigest() == 'eb9058c3a382ffc7106e4002c42a8d85'
        f.write(f_content)
    print("dataset downloaded")
    logging.info("dataset downloaded")
    return target_path


def download_extract_cifar100(root="data"):
    tar_file_path = _download_cifar100(root=root)
    print("extracting CIFAR 100")
    logging.info("extracting dataset")
    if tar_file_path is not None:
        with tarfile.open(tar_file_path, 'r') as tar:
            tar.extractall(root)
    print("dataset extracted")
    logging.info("dataset extracted")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="./data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    download_extract_cifar100(args.dataset_path)