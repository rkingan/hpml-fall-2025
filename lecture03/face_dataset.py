"""

Pytorch Dataset implementation for the face landmarks data set.

"""
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

class FaceLandmarksDataset(Dataset):
    """

    Face landmarks dataset


    """

    def __init__(self,
                 csv_file : str | Path,
                 images_path : str | Path,
                 transform : Callable[[dict], dict]=None
        ):
        """

        Initializes the FaceDataset object.

        Parameters
        ----------

        csv_file : str | Path
            Path to the CSV file containing labels and metadata.
        images_path : str | Path
            Path to the file or directory containing the images.
        transform : callable, optional
            Optional transform to be applied on a sample.

        """
        self.keypoints_frame = pd.read_csv(csv_file)
        self.images_path = Path(images_path)
        self.transform = transform
        self.last_loaded_image_index = -1
        self.last_loaded_image_segment = None
        self._call_count = 0

    def __len__(self):
        """

        Returns the length of the dataset.

        Returns
        -------

        int
            Length of the dataset.

        """
        return len(self.keypoints_frame)


    def _load_image_segment(self, image_index: int):
        if image_index != self.last_loaded_image_index:
            image_segment_path = self.images_path / f"face_images_batch_{image_index + 1}.npz"
            with np.load(image_segment_path) as data:
                self.last_loaded_image_segment = data["images"]
            self.last_loaded_image_index = image_index


    def __getitem__(self, idx):
        """

        Retrieves a sample from the dataset at the specified index.

        Parameters
        ----------

        idx : int
            Index of the sample to retrieve.

        Returns
        -------

        dict
            A dictionary containing the image and its corresponding keypoints.

        """
        self._call_count += 1
        if self._call_count % 500 == 0:
            log.info(f"__getitem__ called %d times in process %d, last index was %d", self._call_count, os.getpid(), idx)

        if isinstance(idx, np.integer):
            idx = int(idx)

        image_index = idx // 1000
        image_offset = idx % 1000
        self._load_image_segment(image_index)
        image = self.last_loaded_image_segment[image_offset]
        keypoints = self.keypoints_frame.iloc[idx, 1:].to_numpy()
        keypoints = keypoints.astype("float")
        sample = {"image": image, "keypoints": keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample