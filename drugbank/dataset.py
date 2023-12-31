import os
import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
from data_preprocessing_cold import CustomData


# %%
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


class DrugDataset(Dataset):
    def __init__(self, data_df, drug_graph):
        self.data_df = data_df
        self.drug_graph = drug_graph

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []

        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            Neg_ID, Ntype = Neg_samples.split('$')
            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)
            n_graph = self.drug_graph.get(Neg_ID)

            pos_pair_h = h_graph
            pos_pair_t = t_graph

            if Ntype == 'h':
                neg_pair_h = n_graph
                neg_pair_t = t_graph
            else:
                neg_pair_h = h_graph
                neg_pair_t = n_graph

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)

            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)

        return head_pairs, tail_pairs, rel, label


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def split_train_valid(data_df, fold, val_ratio=0.2):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['Y'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df


def load_ddi_dataset(root, batch_size, fold=0):
    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_train_fold{fold}.csv'))
    test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
    train_df, val_df = split_train_valid(train_df, fold=fold)

    train_set = DrugDataset(train_df, drug_graph)
    val_set = DrugDataset(val_df, drug_graph)
    test_set = DrugDataset(test_df, drug_graph)
    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader


def load_ddi_dataset_cold(root, batch_size, fold=0):
    filename = 'data/preprocessed/drugbank'
    drug_graph = read_pickle(os.path.join(filename, 'drug_data.pkl'))
    train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets-fold{fold}-train.csv'))
    M_s1 = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets-fold{fold}-s1.csv'))
    M_s2_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets-fold{fold}-s2.csv'))

    train_set = DrugDataset(train_df, drug_graph)
    M_s1_set = DrugDataset(M_s1, drug_graph)
    M_s2_set = DrugDataset(M_s2_df, drug_graph)

    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    M_s1_loader = DrugDataLoader(M_s1_set, batch_size=batch_size, shuffle=False, num_workers=8)
    M_s2_loader = DrugDataLoader(M_s2_set, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the Ms1 set: ", len(M_s1_set))
    print("Number of samples in the Ms2 set: ", len(M_s2_set))

    return train_loader, M_s2_loader, M_s1_loader


if __name__ == "__main__":
    # train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/drugbank', batch_size=256, fold=0)
    train_loader, val_loader, test_loader = load_ddi_dataset_cold(root='data/preprocessed/cold_start/drugbank',
                                                                  batch_size=256, fold=0)

# %%
