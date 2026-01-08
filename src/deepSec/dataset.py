import os
import sys
import pandas as pd
import gzip
import yaml
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .utils import *

class CustomDataset(Dataset):
    def __init__(self, in_file, uniq_seq=False, idx_by_species=True):
        self.out_file = in_file.split('.txt')[0]
        self.X = []
        self.y = []

        if idx_by_species:
            self.train_idx = []
            self.val_idx = []
            self.test_idx = []

        if uniq_seq:
            D = {}
            inFile = gzip.open(in_file, 'rt')
            head = inFile.readline()
            for line in inFile:
                line = line.strip()
                fields = line.split('\t')
                gene_name = fields[2]
                seq = fields[-1]
                cls = fields[-2]
                D[seq] = cls
            inFile.close()

            for k in D:
                self.X.append(k)
                self.y.append(D[k])
        else:
            inFile = gzip.open(in_file, 'rt')
            head = inFile.readline()
            n = -1
            for line in inFile:
                n += 1
                line = line.strip()
                fields = line.split('\t')
                gene_name = fields[2]
                seq = fields[-1]
                cls = fields[-2]
                self.X.append(seq)
                self.y.append(cls)

                if idx_by_species:
                    if fields[0].find('metazoa_') != -1 or fields[0].find('protists_') != -1:
                        self.train_idx.append(n)
                    elif fields[0].find('vertebrate_') != -1:
                        self.val_idx.append(n)
            inFile.close()

    def __len__(self):
        return(len(self.y))

    def __getitem__(self, idx, alphabet=['N', 'A', 'C', 'G', 'T'], alphabet2=['negative', 'positive']):
        X = self.X[idx]
        y = self.y[idx]
        X = torch.tensor([alphabet.index(x) if x in alphabet else 0 for x in X])
        y = torch.tensor(alphabet2.index(y))
        return(X, y)

    def split_save_dataset(self, ratio=[0.8, 0.1, 0.1], split_by_species=True, split_by_species_ratio=[0.5, 0.5], batch_size=32, shuffle=True, num_workers=0):
        self.train_file = self.out_file + '_dataset_train.pt'
        self.val_file = self.out_file + '_dataset_val.pt'
        self.test_file = self.out_file + '_dataset_test.pt'

        if split_by_species:
            self.ds_train = torch.utils.data.Subset(self, self.train_idx)
            self.ds_val = torch.utils.data.Subset(self, self.val_idx)
            self.ds_val, self.ds_test = torch.utils.data.random_split(self.ds_val, [split_by_species_ratio[0], split_by_species_ratio[1]])
        else:
            self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(self, ratio)

        self.dl_train = DataLoader(self.ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.dl_val = DataLoader(self.ds_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.dl_test = DataLoader(self.ds_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        torch.save(self.dl_train, self.train_file)
        torch.save(self.dl_val, self.val_file)
        torch.save(self.dl_test, self.test_file)

class CustomDatasetPredPaired(Dataset):
    def __init__(self, in_file):
        self.in_file = in_file
        self.X0 = []
        self.X1 = []
        self.X2 = []

        self.pred_file = in_file.split('.txt')[0] + '_dataset_pred.pt'

        inFile = gzip.open(in_file, 'rt')
        head = inFile.readline()
        n = -1
        for line in inFile:
            n += 1
            line = line.strip()
            fields = line.split('\t')
            self.X0.append(n)
            self.X1.append(fields[-2])
            self.X2.append(fields[-1])
        inFile.close()

    def __len__(self):
        return(len(self.X1))

    def __getitem__(self, idx, alphabet=['N', 'A', 'C', 'G', 'T']):
        X0 = self.X0[idx]
        X1 = self.X1[idx]
        X2 = self.X2[idx]
        X0 = torch.tensor(X0)
        X1 = torch.tensor([alphabet.index(x) if x in alphabet else 0 for x in X1])
        X2 = torch.tensor([alphabet.index(x) if x in alphabet else 0 for x in X2])
        return(X0, X1, X2)

    def split_save_dataset(self, batch_size=32, shuffle=False, num_workers=0):
        dl_pred = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        torch.save(dl_pred, self.pred_file)


class CustomDatasetPredSingle(Dataset):
    def __init__(self, in_file):
        self.in_file = in_file
        self.X0 = []
        self.X1 = []

        self.pred_file = in_file.split('.txt')[0] + '_dataset_pred.pt'

        inFile = gzip.open(in_file, 'rt')
        head = inFile.readline()
        n = -1
        for line in inFile:
            n += 1
            line = line.strip()
            fields = line.split('\t')
            self.X0.append(n)
            self.X1.append(fields[-1])
        inFile.close()

    def __len__(self):
        return(len(self.X1))

    def __getitem__(self, idx, alphabet=['N', 'A', 'C', 'G', 'T']):
        X0 = self.X0[idx]
        X1 = self.X1[idx]
        X0 = torch.tensor(X0)
        X1 = torch.tensor([alphabet.index(x) if x in alphabet else 0 for x in X1])
        return(X0, X1)

    def split_save_dataset(self, batch_size=32, shuffle=False, num_workers=0):
        dl_pred = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        torch.save(dl_pred, self.pred_file)


if __name__ == '__main__':
    ds = CustomDataset('Ensembl_Protein2Transcript_Upos_FeatureSeq_DownSampling100.txt.gz')
    ds.split_save_dataset()

