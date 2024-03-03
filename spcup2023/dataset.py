import random
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset


class spcup23_ds(Dataset):
    """
    Kaggle: 
        > For the binarized labeling, use 0 for schizophrenia (SZ) and 1 for Bipolar (BP).
    """

    def __init__(self, path, transform = None, test = False, seed=2137, shuffle=False):
        """

        """
        self._seed = seed
        self._shuffle = shuffle
        self._transform = transform

        # path to the dataset directory
        if not isinstance(path, Path):
            path = Path(path)
        self._path = path

        # Create container for dataset items
        self._data = []

        # Load dataset into RAM
        if test:
            self._test_ds2ram()
        else:
            self._train_ds2ram()

    def _train_ds2ram(self):
        """
        """
        BP = (1, self._path / "train" / "BP")
        SZ = (0, self._path / "train" / "SZ")

        for label, path in (BP, SZ): # Iterate each class
            for filename in path.glob("sub*"): # Iterate each dir for given class
                # Load files
                folder = str(filename).split("/")[-1]
                fnc = np.load(filename / "fnc.npy").squeeze()
                icn_tc = np.load(filename / "icn_tc.npy").squeeze()

                # Create tuple containing item form dataset
                item = (folder, label, fnc, icn_tc)

                # Push item to the container
                self._data.append(item)
        
        # Deterministic starting point before shuffle
        self._data.sort(key=lambda x : x[0])  
        
        # Shuffle dataset
        if self._shuffle:
            random.Random(self._seed).shuffle(self._data)

    def _test_ds2ram(self):
        """
        """
        path = self._path / "test"

        for filename in path.glob("sub*"): # Iterate each dir for given class
            # Load files
            folder = str(filename).split("/")[-1]
            fnc = np.load(filename / "fnc.npy").squeeze()
            icn_tc = np.load(filename / "icn_tc.npy").squeeze()

            # Create tuple containing item form dataset
            item = (folder, -1, fnc, icn_tc)

            # Push item to the container
            self._data.append(item)
        
        # Deterministic starting point before shuffle
        self._data.sort(key=lambda x : x[0])
        
        # Shuffle dataset
        if self._shuffle:
            random.Random(self._seed).shuffle(self._data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self):
            result = self[self._n]
            self._n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx):
        """
        """
        filename, label, fnc, icn_tc = self._data[idx]
        sample = (filename, label, fnc.copy(), icn_tc.copy())
        if self._transform:
            sample = self._transform(sample)

        return sample