from operator import index
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import random

'''
The D-MPNN takes line graphs instead of node graphs used in common GNN as input. 
This is because the D-MPNN operates on edges/bonds instead of nodes. 
So we have to convert the node graph to the line graph.
'''


class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''

    def __inc__(self, key, value, *args, **kwargs):
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"


def one_of_k_encoding(k, possible_values):
    '''
    Convert integer to one-hot representation.
    '''
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    '''
    Convert integer to one-hot representation.
    '''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    '''
    Get atom features. Note that atom.GetFormalCharge() can return -1
    '''
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def edge_features(bond):
    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def generate_drug_data(mol_graph, atom_symbols):
    # (bond_i, bond_j, bond_features)
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    # Separate (bond_i, bond_j, bond_features) to (bond_i, bond_j) and bond_features
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    # Convert the graph to undirect graph, e.g., [(1, 0)] to [(1, 0), (0, 1)]
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    # This is the most essential step to convert a node graph to a line graph
    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T

    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index,
                      edge_attr=edge_feats)

    return data


def load_drug_mol_data(args):
    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}

    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1],
                                                    data[args.c_s2], data[args.c_y]):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2

    for id, smiles in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol_tup.append((id, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

    symbols = list(set(symbols))
    drug_data = {id: generate_drug_data(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    save_data(drug_data, 'drug_data.pkl', args)
    return drug_data


def generate_pair_triplets(args):
    pos_triplets = []
    drug_ids = []

    with open(f'{args.dirname}/{args.dataset.lower()}/drug_data.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_y]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        # Drugbank dataset is 1-based index, need to substract by 1
        if args.dataset in ('drugbank',):
            relation -= 1
        pos_triplets.append([id1, id2, relation])

    if len(pos_triplets) == 0:
        raise ValueError('All tuples are invalid.')

    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    neg_samples = []
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        temp_neg = []
        h, t, r = pos_item[:3]

        if args.dataset == 'drugbank':
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                       [str(neg_t) + '$t' for neg_t in neg_tails]
        else:
            existing_drug_ids = np.asarray(list(set(
                np.concatenate(
                    [data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]],
                    axis=0)
            )))
            temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)

        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))

    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0],
                       'Drug2_ID': pos_triplets[:, 1],
                       'Y': pos_triplets[:, 2],
                       'Neg samples': neg_samples})
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df.to_csv(filename, index=False)
    print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


def generate_pair_triplets_cold(args):

    n_folds = args.n_folds

    for fold_i in range(n_folds):
        pos_triplets_Ms1 = []
        pos_triplets_Ms2 = []
        pos_triplets_train = []
        pos_triplets_Ms1_ids = set()
        pos_triplets_Ms2_ids = set()
        with open(f'{args.dirname}/{args.dataset.lower()}/drug_data.pkl', 'rb') as f:

            drug_ids = list(pickle.load(f).keys())
            drug_ids_cold = random.sample(drug_ids, 342)
            drug_ids_train = list(set(drug_ids) - set(drug_ids_cold))
        # 读入指定的数据文件drugbank.tab或twosides_ge_500.csv，将其转化为一个 pandas DataFrame 对象=id1，id2，Y，map，X1，X2（smlies）共6项指标
        data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
        # 逐个遍历其中每一行数据，将有效的实体对及其关系存储到列表 pos_triplets（正例三元组） 中（如果 id1 或 id2 不在药品 ID 中则视为无效记录）。
        # Drugbank dataset is 1-based index, need to substract by 1，以补偿数组下标与实际编号的差别。
        for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_y]):
            # 如果 id1 或 id2 不在总药品 ID 中，则说明该条记录无效，跳过当前循环，继续处理下一行三元组数据。
            if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
            # 如果id1 与 id2 都在冷药品 ID 中，则加入Ms1
            if ((id1 in drug_ids_cold) and (id2 in drug_ids_cold)):
                relation -= 1
                pos_triplets_Ms1.append([id1, id2, relation])
                pos_triplets_Ms1_ids.add(id1)
                pos_triplets_Ms1_ids.add(id2)
                continue
            if ((id1 not in drug_ids_cold) and (id2 not in drug_ids_cold)):
                relation -= 1
                pos_triplets_train.append([id1, id2, relation])
                continue
            relation -= 1
            # 将有效的实体对及其关系依次追加到正例三元组列表 pos_triplets中。Drugbank：191808组
            pos_triplets_Ms2.append([id1, id2, relation])
            pos_triplets_Ms2_ids.add(id1)
            pos_triplets_Ms2_ids.add(id2)

        drug_ids_Ms1 = list(pos_triplets_Ms1_ids.intersection(set(drug_ids_cold)))
        drug_ids_Ms2 = list(pos_triplets_Ms2_ids.intersection(set(drug_ids_cold)))

        if len(pos_triplets_train) == 0 or len(pos_triplets_Ms1) == 0 or len(pos_triplets_Ms2) == 0:
            raise ValueError('At least one of the lists is empty.')
        # 将 pos_triplets 与 drug_ids 从 list 转化为 numpy 的 array 对象，并调用函数 load_data_statistics
        # 计算该三元组数组中每个关系对应的一些常见信息，返回结果为一个字典类型的变量 data_statistics。
        pos_triplets_train = np.array(pos_triplets_train)
        pos_triplets_Ms1 = np.array(pos_triplets_Ms1)
        pos_triplets_Ms2 = np.array(pos_triplets_Ms2)
        data_statistics_train = load_data_statistics_cold(pos_triplets_train, fold_i, name='train')
        data_statistics_Ms1 = load_data_statistics_cold(pos_triplets_Ms1, fold_i, name='Ms1')
        data_statistics_Ms2 = load_data_statistics_cold(pos_triplets_Ms2, fold_i, name='Ms2')
        drug_ids = np.array(drug_ids)
        drug_ids_cold = np.array(drug_ids_cold)
        drug_ids_train = np.array(drug_ids_train)
        drug_ids_Ms1 = np.array(drug_ids_Ms1)
        drug_ids_Ms2 = np.array(drug_ids_Ms2)

        neg_samples_train = []
        neg_samples_Ms1 = []
        neg_samples_Ms2 = []
        for pos_item in tqdm(pos_triplets_train, desc=f'Generating  TrainSet{fold_i} Negative sample'):
            temp_neg = []
            # 将列表 pos_item 的前三个元素分别赋值给变量 h、 t 和 r
            h, t, r = pos_item[:3]
            # 在 drugbank 数据集中，通过 _normal_batch 函数使用头尾分裂策略得到负例，并将其转化为字符串格式添加到 temp_neg 列表中；
            # 调用函数 _normal_batch 来随机生成负头/尾实体，并按照 <实体 ID>$h 和 <实体 ID>$t 的形式拼接成一个字符串列表 temp_neg。
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics_train, drug_ids_train, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                       [str(neg_t) + '$t' for neg_t in neg_tails]

            # 将生成的负例数据temp_neg转换为字符串格式，并使用下划线连接每个元素，从而得到一个以下划线分隔的字符串;
            # map函数将temp_neg[:args.neg_ent]中的每个数字都映射为字符串类型
            neg_samples_train.append('_'.join(map(str, temp_neg[:args.neg_ent])))
        # 使用 pd.DataFrame 构造函数创建了一个 DataFrame 对象 df。其中，'Drug1_ID', 'Drug2_ID', 'Y', 'Neg samples' 分别为 df 的列名；pos_triplets 数组的前三列被赋值给 'Drug1_ID', 'Drug2_ID', 'Y' 列；neg_samples 列表被赋给 'Neg samples' 列。这样构造出来的 df 具有四列：两个药物实体的 ID，一个关系类型以及相应的负样本。
        df = pd.DataFrame({'Drug1_ID': pos_triplets_train[:, 0],
                           'Drug2_ID': pos_triplets_train[:, 1],
                           'Y': pos_triplets_train[:, 2],
                           'Neg samples': neg_samples_train})
        filename_cold = f'{args.dirname}/cold_start/drugbank'
        if not os.path.exists(filename_cold):
            os.makedirs(filename_cold)
        filename = f'{args.dirname}/cold_start/{args.dataset}/pair_pos_neg_triplets-fold{fold_i}-train.csv'
        # 使用 to_csv() 方法将 df 存储为一个 CSV 文件,index=False 参数表示不将行索引写入 CSV 文件中。
        df.to_csv(filename, index=False)
        print(f'\nData saved as {filename}!')
        save_data_cold(data_statistics_train, f'data_statistics_train{fold_i}.pkl', args)

        for pos_item in tqdm(pos_triplets_Ms1, desc='Generating  Ms1Set Negative sample'):
            temp_neg = []
            # 将列表 pos_item 的前三个元素分别赋值给变量 h、 t 和 r
            h, t, r = pos_item[:3]
            # 在 drugbank 数据集中，通过 _normal_batch 函数使用头尾分裂策略得到负例，并将其转化为字符串格式添加到 temp_neg 列表中；
            # 调用函数 _normal_batch 来随机生成负头/尾实体，并按照 <实体 ID>$h 和 <实体 ID>$t 的形式拼接成一个字符串列表 temp_neg。
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics_Ms1, drug_ids_Ms1, args)
            # GMPNN冷启动负例生成
            # neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics_Ms1, drug_ids_cold, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                       [str(neg_t) + '$t' for neg_t in neg_tails]

            # 将生成的负例数据temp_neg转换为字符串格式，并使用下划线连接每个元素，从而得到一个以下划线分隔的字符串;
            # map函数将temp_neg[:args.neg_ent]中的每个数字都映射为字符串类型
            neg_samples_Ms1.append('_'.join(map(str, temp_neg[:args.neg_ent])))
        # 使用 pd.DataFrame 构造函数创建了一个 DataFrame 对象 df。其中，'Drug1_ID', 'Drug2_ID', 'Y', 'Neg samples' 分别为 df 的列名；pos_triplets 数组的前三列被赋值给 'Drug1_ID', 'Drug2_ID', 'Y' 列；neg_samples 列表被赋给 'Neg samples' 列。这样构造出来的 df 具有四列：两个药物实体的 ID，一个关系类型以及相应的负样本。
        df = pd.DataFrame({'Drug1_ID': pos_triplets_Ms1[:, 0],
                           'Drug2_ID': pos_triplets_Ms1[:, 1],
                           'Y': pos_triplets_Ms1[:, 2],
                           'Neg samples': neg_samples_Ms1})
        filename = f'{args.dirname}/cold_start/{args.dataset}/pair_pos_neg_triplets-fold{fold_i}-s1.csv'
        # 使用 to_csv() 方法将 df 存储为一个 CSV 文件,index=False 参数表示不将行索引写入 CSV 文件中。
        df.to_csv(filename, index=False)
        print(f'\nData saved as {filename}!')
        save_data_cold(data_statistics_Ms1, f'data_statistics_Ms1{fold_i}.pkl', args)

        for pos_item in tqdm(pos_triplets_Ms2, desc='Generating  Ms2Set Negative sample'):
            temp_neg = []
            # 将列表 pos_item 的前三个元素分别赋值给变量 h、 t 和 r
            h, t, r = pos_item[:3]
            # 在 drugbank 数据集中，通过 _normal_batch 函数使用头尾分裂策略得到负例，并将其转化为字符串格式添加到 temp_neg 列表中；
            # 调用函数 _normal_batch 来随机生成负头/尾实体，并按照 <实体 ID>$h 和 <实体 ID>$t 的形式拼接成一个字符串列表 temp_neg。
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics_Ms2, drug_ids_Ms2, args)
            # GMPNN冷启动负例生成
            # neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics_Ms2, drug_ids_cold, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                       [str(neg_t) + '$t' for neg_t in neg_tails]

            # 将生成的负例数据temp_neg转换为字符串格式，并使用下划线连接每个元素，从而得到一个以下划线分隔的字符串;
            # map函数将temp_neg[:args.neg_ent]中的每个数字都映射为字符串类型
            neg_samples_Ms2.append('_'.join(map(str, temp_neg[:args.neg_ent])))
        # 使用 pd.DataFrame 构造函数创建了一个 DataFrame 对象 df。其中，'Drug1_ID', 'Drug2_ID', 'Y', 'Neg samples' 分别为 df 的列名；pos_triplets 数组的前三列被赋值给 'Drug1_ID', 'Drug2_ID', 'Y' 列；neg_samples 列表被赋给 'Neg samples' 列。这样构造出来的 df 具有四列：两个药物实体的 ID，一个关系类型以及相应的负样本。
        df = pd.DataFrame({'Drug1_ID': pos_triplets_Ms2[:, 0],
                           'Drug2_ID': pos_triplets_Ms2[:, 1],
                           'Y': pos_triplets_Ms2[:, 2],
                           'Neg samples': neg_samples_Ms2})
        filename = f'{args.dirname}/cold_start/{args.dataset}/pair_pos_neg_triplets-fold{fold_i}-s2.csv'
        # 使用 to_csv() 方法将 df 存储为一个 CSV 文件,index=False 参数表示不将行索引写入 CSV 文件中。
        df.to_csv(filename, index=False)
        print(f'\nData saved as {filename}!')
        save_data_cold(data_statistics_Ms2, f'data_statistics_Ms2{fold_i}.pkl', args)


def load_data_statistics_cold(all_tuples, fold_i, name):
    '''
    统计给定正例三元组数据集中每个关系类型的各种信息，并将其存储到一个字典中,7个键。
    :param all_tuples:正样本三元组数组
    :return:字典
    '''
    print(f'Loading data statistics cold_start {name} fold {fold_i}...')
    # 定义一个名为 statistics 的字典，包括7个键；H-R-T
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    # 创建了一个名为 statistics["ALL_H_WITH_R"] 的空字典，默认情况下，尝试访问该字典中不存在的键时会返回一个空的字典。
    # 也就是说，可以使用类似 statistics["ALL_H_WITH_R"][r][h] = value 的语句向 statistics["ALL_H_WITH_R"] 中加入键值对。
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}
    # 对于三元组(h,t,r)，将 h、t、r 的信息加入到字典 statistics 的相应标记下。
    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        # 将 h 添加到字典 statistics["ALL_TRUE_H_WITH_TR"][(t, r)] 对应的列表值中，以便于在 (t, r) 给定时快速找到所有真正的头实体。
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        # 给字典 statistics["FREQ_REL"] 中与关系类型 r 相关的值加 1，以此计算关系类型 r 在整个数据集中出现的频率。drugbank=86
        statistics["FREQ_REL"][r] += 1.0
        # 将包含特定关系类型 r 的所有头实体及对应的标记，以字典的形式存储在 statistics["ALL_H_WITH_R"][r] 中。
        # 对于 statistics["ALL_H_WITH_R"][r]，它创建一个字典，并将 h 作为键并将其相应的值设定为 1。
        # 如果之后还有其他三元组的关系类型为 r，且他们的头实体也为 h，那么就会通过该行代码对同一字典进行多次赋值，从而持续更新键值对。
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1
    # 对于 "ALL_TRUE_H_WITH_TR" 和 "ALL_TRUE_T_WITH_HR" 中的每一个键值对，去重H/T并转换为 numpy 数组。
    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    # 循环遍历包含所有真实尾实体的三元组 (h,t,r) 的列表，使用 (h, r) 作为键将对应的值取出，利用 set() 函数去除其中的重复元素，并将结果转化为 NumPy 数组。
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))
    # 对三元组中所有关系类型，
    for r in statistics["FREQ_REL"]:
        # 针对每个关系r，将所有出现过的头/尾实体分别存入一个 NumPy 数组中，并覆盖原来的键值。也就是说，该指标统计了每个关系r对应的所有头/尾实体。
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        # 针对每个关系 r，用该关系在训练数据中出现的次数除以该关系对应的所有尾实体数量求得一个比例，并保存到 statistics["ALL_HEAD_PER_TAIL"][r] 中。
        # 也就是说，该指标用来反映相同关系下，去重前头实体可能的个数与去重后尾实体个数之间的比例关系。计算该关系 r 的头实体到尾实体的单向比例。
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        # 该指标用来反映相同关系下，去重前尾实体可能的个数与去重后头实体个数之间的比例关系。
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])

    print(f'Getting data statistics cold_start {name} fold {fold_i} done!')

    return statistics


def save_data_cold(data, filename, args):
    '''
    保存药物特征表示至指定目录
    :param data: 要保存的数据
    :param filename:要保存的文件名
    :param args:命令行参数
    :return:
    '''
    # 生成目录名
    dirname = f'{args.dirname}/cold_start/{args.dataset}'
    # 如果这个目录不存在则创建它。
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # 将目标文件名设置为目录名加上文件名，放在变量 filename 中
    filename = dirname + '/' + filename
    # 使用 pickle.dump() 将 data 对象序列化到文件中，以二进制格式写入。最后打印出已经保存的文件名，提示保存成功。
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def load_data_statistics(all_tuples):
    '''
    This function is used to calculate the probability in order to generate a negative.
    You can skip it because it is unimportant.
    '''
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])

    print('getting data statistics done!')

    return statistics


def _corrupt_ent(positive_existing_ents, max_num, drug_ids, args):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = args.random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])

    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents


def _normal_batch(h, t, r, neg_size, data_statistics, drug_ids, args):
    neg_size_h = 0
    neg_size_t = 0
    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] +
                                                      data_statistics["ALL_HEAD_PER_TAIL"][r])
    # prob = 2
    for i in range(neg_size):
        if args.random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t += 1

    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids, args),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids, args))


def save_data(data, filename, args):
    dirname = f'{args.dirname}/{args.dataset}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def split_data(args):
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df = pd.read_csv(filename)
    seed = args.seed
    class_name = args.class_name
    test_size_ratio = args.test_ratio
    n_folds = args.n_folds
    save_to_filename = os.path.splitext(filename)[0]
    cv_split = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size_ratio, random_state=seed)
    for fold_i, (train_index, test_index) in enumerate(cv_split.split(X=df, y=df[class_name])):
        print(f'Fold {fold_i} generated!')
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df.to_csv(f'{save_to_filename}_train_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_train_fold{fold_i}.csv', 'saved!')
        test_df.to_csv(f'{save_to_filename}_test_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_test_fold{fold_i}.csv', 'saved!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['drugbank', 'twosides'],
                        help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str, required=True,
                        choices=['all', 'generate_triplets', 'drug_data', 'split'], help='Operation to perform')
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)

    dataset_columns_map = {
        'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    }

    dataset_file_name_map = {
        'drugbank': ('data/drugbank.tab', '\t'),
        'twosides': ('data/twosides_ge_500.csv', ',')
    }
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    args.dirname = 'data/preprocessed'

    args.random_num_gen = np.random.RandomState(args.seed)
    if args.operation in ('all', 'drug_data'):
        load_drug_mol_data(args)

    if args.operation in ('all', 'generate_triplets'):
        generate_pair_triplets(args)
        generate_pair_triplets_cold(args)

    if args.operation in ('all', 'split'):
        args.class_name = 'Y'
        split_data(args)
