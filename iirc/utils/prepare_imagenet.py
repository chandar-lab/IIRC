import argparse
import subprocess
from scipy.io import loadmat
import time
import os
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
                        help="The folder where the ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_train.tar and "
                             "ILSVRC2012_img_val.tar exist")
    parser.add_argument('--target_dir', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    slurm_tmpdir = os.environ['SLURM_TMPDIR']
    to_be_zipped_folder = os.path.join(slurm_tmpdir, "imagenet_prepared")
    os.makedirs(os.path.join(to_be_zipped_folder), exist_ok=True)

    if os.path.exists(os.path.join(args.target_dir, "imagenet_prepared.tar")):
        logging.info(f"imagenet_prepared.tar already exists")
    else:
        start_time = time.time()
        assert os.path.exists(os.path.join(args.source_dir, "ILSVRC2012_devkit_t12.tar.gz"))
        assert os.path.exists(os.path.join(args.source_dir, "ILSVRC2012_img_train.tar"))
        assert os.path.exists(os.path.join(args.source_dir, "ILSVRC2012_img_val.tar"))

        logging.info(f"copying files")
        subprocess.run(f"rsync --progress -av {os.path.join(args.source_dir, '*')} {slurm_tmpdir}", shell=True)
        logging.info(f"Copying done")

        logging.info(f"unzipping ILSVRC2012_devkit_t12.tar.gz")
        subprocess.run(f"tar -xf {os.path.join(slurm_tmpdir, 'ILSVRC2012_devkit_t12.tar.gz')} "
                       f"-C {to_be_zipped_folder}", shell=True)

        meta = loadmat(os.path.join(to_be_zipped_folder, 'ILSVRC2012_devkit_t12', 'data', 'meta.mat'))
        meta = meta["synsets"]
        ilsvrc_id_2_wnid = {int(synset[0][0][0]): str(synset[0][1][0]) for synset in meta}

        val_ground_truth_file = os.path.join(to_be_zipped_folder, 'ILSVRC2012_devkit_t12', 'data',
                                             'ILSVRC2012_validation_ground_truth.txt')
        with open(val_ground_truth_file, "r") as f:
            val_ground_truth = f.readlines()
        val_ground_truth = [int(id.strip('\n')) for id in val_ground_truth]
        ilsvrc_ids = list(set(val_ground_truth))
        ilsvrc_id_2_wnid_keys = list(ilsvrc_id_2_wnid.keys())
        for id in ilsvrc_id_2_wnid_keys:
            if id not in ilsvrc_ids:
                del ilsvrc_id_2_wnid[id]

        os.makedirs(os.path.join(to_be_zipped_folder, "train"), exist_ok=True)
        logging.info(f"unzipping ILSVRC2012_img_train.tar")
        subprocess.run(f"tar -xf {os.path.join(slurm_tmpdir, 'ILSVRC2012_img_train.tar')} "
                       f"-C {os.path.join(to_be_zipped_folder, 'train')}", shell=True)
        for wnid in ilsvrc_id_2_wnid.values():
            os.makedirs(os.path.join(to_be_zipped_folder, "train", wnid), exist_ok=True)
            logging.info(f"unzipping {wnid}.tar")
            subprocess.run(f"tar -xf {os.path.join(to_be_zipped_folder, 'train', wnid + '.tar')} "
                           f"-C {os.path.join(to_be_zipped_folder, 'train', wnid)}", shell=True)

        os.makedirs(os.path.join(to_be_zipped_folder, "val"), exist_ok=True)
        logging.info(f"unzipping ILSVRC2012_img_val.tar")
        subprocess.run(f"tar -xf {os.path.join(slurm_tmpdir, 'ILSVRC2012_img_val.tar')} "
                       f"-C {os.path.join(to_be_zipped_folder, 'val')}", shell=True)
        val_files = [os.path.join(to_be_zipped_folder, "val", f"ILSVRC2012_val_{i:08d}.JPEG") for i in range(1, 50001)]
        for id, wnid in ilsvrc_id_2_wnid.items():
            logging.info(f"Separating the validation samples for class {wnid}")
            os.makedirs(os.path.join(to_be_zipped_folder, "val", wnid), exist_ok=True)
            command = f"mv -t {os.path.join(to_be_zipped_folder, 'val', wnid)}"
            count = 0
            for i in range(50000):
                if val_ground_truth[i] == id:
                    command += f" {val_files[i]}"
                    count += 1
            assert count == 50
            subprocess.run(command, shell=True)

        logging.info(f"removing the training per-class tar files")
        subprocess.run(f"rm {os.path.join(to_be_zipped_folder, 'train', '*.tar')}", shell=True)

        logging.info(f"archiving and compressing imagenet_prepared.tar.gz")
        subprocess.run(f"tar -zcf {os.path.join(slurm_tmpdir, 'imagenet_prepared.tar')} -C {slurm_tmpdir} "
                       f"imagenet_prepared", shell=True)

        logging.info(f"putting imagenet_prepared.tar in {args.target_dir}")
        subprocess.run(f"rsync --progress {os.path.join(slurm_tmpdir, 'imagenet_prepared.tar')} "
                       f"{args.target_dir}", shell=True)
