import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import LGConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from tqdm.auto import tqdm, trange


class LightGCNLayer(MessagePassing):
    def __init__(self, norm):
        super().__init__(aggr='add')
        self.norm = norm

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=self.norm)

    def message(self, x_j):
        return self.norm.view(-1, 1) * x_j


class LightGCN(nn.Module):

    def __init__(self, num_users, num_items, emb_dim, num_layers, edge_index, user_item_matrix, device):
        super().__init__()
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items
        self.user_item_matrix = user_item_matrix
        self.device = device
        num_nodes = num_users + num_items
        self.embeddings = nn.Embedding(num_nodes, emb_dim, device=device)
        self.norm = self.precompute_norm(edge_index, num_nodes)
        self.layers = nn.ModuleList([LightGCNLayer(self.norm) for _ in range(num_layers)])
        self.edge_index = edge_index

    def forward(self, x):
        x = self.embeddings(x)
        all_embeddings = [x]
        for layer in self.layers:
            x = layer(x, edge_index=self.edge_index)
            all_embeddings.append(x)
        return (sum(all_embeddings) / self.num_layers)

    @property
    def user_embeddings(self):
        return self.embeddings.weight[:self.num_users]

    @property
    def item_embeddings(self):
        return self.embeddings.weight[self.num_users:]

    @staticmethod
    def precompute_norm(edge_index, num_nodes):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm
        # return SparseTensor(row=row, col=col, value=norm, sparse_sizes=(num_nodes, num_nodes))

    def fit(self, train_loader, epochs, lr):
        """
        Train the LightGCN model
        Args:
        train_loader: DataLoader for the training set
        epochs: Number of training epochs
        lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in trange(epochs, desc="Epoch", dynamic_ncols=True):
            total_loss = 0
            for user_ids, pos, neg in tqdm(train_loader,
                                           desc="Iteration",
                                           leave=False,
                                           dynamic_ncols=True):
                optimizer.zero_grad()

                # Forward pass
                user_embeddings = self(user_ids.to(self.device))
                positive_item_embeddings = self(pos.to(self.device))
                negative_item_embeddings = self(neg.to(self.device))

                # BPR Loss
                pos_scores = torch.sum(user_embeddings * positive_item_embeddings, dim=1)
                neg_scores = torch.sum(user_embeddings * negative_item_embeddings, dim=1)
                loss = bpr_loss(pos_scores, neg_scores)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    @torch.no_grad()
    def predict(self, user_ids):
        """
        Predict item scores for the given user_ids.
        Args:
        user_ids (Tensor): Tensor of user IDs
        """
        scores = torch.mm(self.embeddings(user_ids), self.item_embeddings.t())
        for idx, user_id in enumerate(user_ids):
            train_items = self.user_item_matrix[user_id].nonzero()[1]
            scores[idx, train_items] = float('-inf')

    @torch.no_grad()
    def validate(self, val_set, k=10):
        """
        Validate the LightGCN model
        Args:
        val_set: DataFrame containing validation interactions
        user_item_matrix: A matrix or DataFrame indicating user-item interactions
        k: Number of top recommendations to consider for metrics
        """
        # Convert DataFrame to torch tensor
        user_ids = torch.tensor(range(val_set['user_id'].nunique()), dtype=torch.long, device=self.device)

        # Predict scores for each user
        self.eval()
        predictions = self.predict(user_ids)

        precisions, recalls, f1s, ndcgs = [], [], [], []

        for user_id in tqdm(user_ids, desc='validation', dynamic_ncols=True):
            # Get the true positive items for this user
            true_items = set(self.user_item_matrix[user_id.item()].nonzero()[1].tolist())

            # Predict top k items
            scores = predictions[user_id]
            # top_k_items = heapq.nlargest(k, torch.range(start=0, end=len(scores)), scores.take)
            probs, top_k_items = torch.topk(scores, k=k)

            # True positives among the top-k recommended items
            true_positives = len(set(top_k_items) & true_items)

            # Compute metrics
            precision = true_positives / k
            recall = true_positives / len(true_items) if true_items else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            ndcg = ndcg_score(top_k_items, true_items, k)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            ndcgs.append(ndcg)

        # Average metrics over all users
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        avg_ndcg = sum(ndcgs) / len(ndcgs)

        return {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1, 'ndcg': avg_ndcg}


class InteractionDataset(Dataset):
    def __init__(self, df, device):
        self.data = df
        self.n_users = df.user_id.nunique()
        self.user_mapping = {user: i for i, user in enumerate(self.data.user_id.unique())}
        self.item_mapping = {item: i for i, item in enumerate(self.data.item_id.unique(), start=self.n_users)}
        self.data.user_id = self.data.user_id.map(self.user_mapping)
        self.data.item_id = self.data.item_id.map(self.item_mapping)
        self.user_ids = self.data.user_id.unique()
        self.item_ids = self.data.item_id.unique()
        self.edge_index = torch.tensor([self.data.user_id, self.data.item_id], dtype=torch.long, device=device)

        self.user_to_positive_items = {user: {'set': set(), 'list': []} for user in self.user_ids}
        self.user_item_matrix = self.create_user_item_matrix()
        for (u, i) in df[['user_id', 'item_id']].values:
            self.user_to_positive_items[u]['set'].add(i)
            self.user_to_positive_items[u]['list'].append(i)

    def create_user_item_matrix(self):
        """
        Create a sparse user-item matrix
        """
        rows = self.data['user_id']
        cols = self.data['item_id'] - self.n_users
        data = [1] * len(self.data)
        return csr_matrix((data, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        positive_sample = random.choice(self.user_to_positive_items[user_id]['list'])

        non_interacted_items = set(self.item_ids) - self.user_to_positive_items[user_id]['set']
        negative_sample = random.choice(list(non_interacted_items))

        return user_id, positive_sample, negative_sample


def bpr_loss(positive_predictions, negative_predictions):
    """
    Bayesian Pairwise Ranking (BPR) Loss
    Args:
    positive_predictions: Predicted scores for positive interactions
    negative_predictions: Predicted scores for negative interactions
    """
    differences = positive_predictions - negative_predictions
    return -torch.sum(F.logsigmoid(differences))


def create_minimal_set(df):
    """
    Create a minimal set of interactions that includes each user and item at least once.
    """
    minimal_set = pd.DataFrame(columns=['user_id', 'item_id'])
    users = df['user_id'].unique()
    items = df['item_id'].unique()

    for user in tqdm(users, desc='users', dynamic_ncols=True):
        user_items = df[df['user_id'] == user]['item_id'].tolist()
        minimal_set = minimal_set.append({'user_id': user, 'item_id': user_items[0]}, ignore_index=True)
        items = list(set(items) - set(user_items[:1]))

    for item in tqdm(items, desc='items', dynamic_ncols=True):
        item_users = df[df['item_id'] == item]['user_id'].tolist()
        minimal_set = minimal_set.append({'user_id': item_users[0], 'item_id': item}, ignore_index=True)

    return minimal_set


def split_dataset(df, minimal_set):
    """
    Split the remaining interactions into train, validation, and test sets.
    """
    remaining_interactions = df[~df.index.isin(minimal_set.index)]
    train, val_test = train_test_split(remaining_interactions, test_size=0.2)
    val, test = train_test_split(val_test, test_size=0.5)

    # Combine minimal set with each split
    train = pd.concat([train, minimal_set]).drop_duplicates().reset_index(drop=True)
    val = pd.concat([val, minimal_set]).drop_duplicates().reset_index(drop=True)
    test = pd.concat([test, minimal_set]).drop_duplicates().reset_index(drop=True)

    return train, val, test


def ndcg_score(model_predictions, true_labels, k=10):
    """
    Compute NDCG score for the model's predictions
    """
    DCG = 0
    IDCG = sum([1.0 / np.log(i + 2) for i in range(min(len(true_labels), k))])
    for i in range(k):
        if i < len(model_predictions) and model_predictions[i] in true_labels:
            DCG += 1.0 / np.log(i + 2)
    return DCG / IDCG if IDCG > 0 else 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_model(batch_size, device):

    df = pd.read_table(
        "/local/scratch/svolokh/TextGCN/data/books/small/train.tsv"
    ).rename(columns={'asin': 'item_id'}).drop_duplicates()
    # val_set = pd.read_table("../data/books/small/test.tsv").rename(columns={'asin': 'item_id'})
    # minimal_set = create_minimal_set(df)
    # train_set, val_set, test_set = split_dataset(df, minimal_set)
    dataset = InteractionDataset(df, device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_layers = 3
    model = LightGCN(
        num_users=df.user_id.nunique(),
        num_items=df.item_id.nunique(),
        emb_dim=64,
        num_layers=num_layers,
        edge_index=dataset.edge_index,
        user_item_matrix=dataset.user_item_matrix,
        device=device,
    )
    return model, train_loader


def main():
    seed_everything(42)
    # device = torch.device('cuda:7')
    device = torch.device('cpu')
    batch_size = 1

    model, loader = load_data_model(batch_size=batch_size, device=device)
    model.fit(loader, 10, 0.01)
    # k = 20
    # for metric, values in model.validate(val_set, dataset.user_item_matrix, k=20).items():
    #     print(f'{metric}@{k}: {values:.4f}')


if __name__ == '__main__':
    main()

"IndexError: Encountered an index error. Please ensure that all indices in 'edge_index' point to valid indices in the interval [0, 0] (got interval [0, 435])"