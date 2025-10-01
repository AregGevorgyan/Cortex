import numpy as np

__all__ = ["Utils"]

class Utils:
    @staticmethod
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    @staticmethod
    def shuffle(data, labels):
        p = np.random.permutation(len(data))
        return data[p], labels[p]
    
    @staticmethod
    def train_test_split(data, labels, test_size=0.2):
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:], labels[:split_idx], labels[split_idx:]