import numpy as np

def split_train_test_val(n_samples, test_ratio=0.1, val_ratio=0.1, seed=0):
    """
    Returns masks to split the samples into train / test / validation sets.

    Returns
    -------
    train: np.ndarray
        Boolean mask
    test: np.ndarray
        Boolean mask
    validation: np.ndarray
        Boolean mask
    """
    if seed:
        np.random.seed(seed)

    # test
    index = np.random.choice(n_samples, int(np.ceil(n_samples * test_ratio)), replace=False)
    test = np.zeros(n_samples, dtype=bool)
    test[index] = 1

    # validation
    index = np.random.choice(np.argwhere(~test).ravel(), int(np.ceil(n_samples * val_ratio)), replace=False)
    val = np.zeros(n_samples, dtype=bool)
    val[index] = 1

    # train
    train = np.ones(n_samples, dtype=bool)
    train[test] = 0
    train[val] = 0
    return train, test, val