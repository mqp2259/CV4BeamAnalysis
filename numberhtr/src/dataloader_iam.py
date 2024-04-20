import pickle
import random
from collections import namedtuple
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 name: str,
                 data_split: float = 0.95,
                 fast: bool = True) -> None:
        """Loader for dataset."""

        assert data_dir.exists()

        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / name / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []
        
        chars = {'-', 'r', 'B', '"', 's', '9', 'X', 'C', 'y', 'A', 'm', ')', 'Z', '8', 'Q', '(', ',', 'H', 'i', '.', 'e', 'P', '1', '/', 'U', 'w', 'l', '2', '+', '?', 'f', '0', 'N', 'Y', '*', 'J', 'M', ':', 'c', 'S', '#', 'O', 'D', 'K', 'd', 'G', 'k', ';', 'a', 'x', 'I', '7', 'g', 'h', 'z', 'b', '5', 'u', ' ', '!', 't', '3', 'T', "'", 'W', 'q', 'o', '4', 'n', 'E', 'p', 'R', 'L', 'F', 'V', 'j', 'v', '&', '6'}
        f = open(data_dir / name / "labels/labels.txt")
        i = 0
        for line in f:
            i += 1
            line = line.strip()
            file_name = data_dir / name / "images" / (str(i) + ".png")
            gt_text = str(line)
            self.samples.append(Sample(gt_text, file_name))
       
        if name  == "train":
            self.train_samples = self.samples
            self.validation_samples = []
        else:
            self.train_samples = []
            self.validation_samples = self.samples

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        if self.fast:
            with self.env.begin() as txn:
                basename = Path(self.samples[i].file_path).basename()
                data = txn.get(basename.encode("ascii"))
                img = pickle.loads(data)
        else:
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))
