import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y, 0)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # Grow the decision tree and return it
        v, c = np.unique(y, return_counts=True)

        predi = v[np.argmax(c)]

        #print(f'{predi}pre, {self._entropy(y):.4f}e, {depth}dep')
        if depth == self.max_depth or self._entropy(y) < 1e-3 or len(y)<5:
            self.progress.update(1)
            return {
                'type': 'leaf',
                'prediction': predi
            }
        
        bf, bt = self._best_split(X, y)
    
        if bf == -1:
            #print(f'bf d{depth} p{predi}')
            self.progress.update(1)
            return {
                'type': 'leaf',
                'prediction': predi
            }
        fn = X.columns[bf]
        lm = X[fn] <= bt
        rm = X[fn] > bt

        lt = self._build_tree(X[lm], y[lm], depth+1)
        rt = self._build_tree(X[rm], y[rm], depth+1)

        self.progress.update(1)
        return{
            'type': 'decision node',
            'feature_idx': bf,
            'threshold': bt,
            'left_tree': lt, 
            'right_tree': rt
        }

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        labels = []
        for x in X.itertuples(index=False):
            x = np.array(x)
            l = self._predict_tree(x, self.tree)
            labels.append(l)

        labels = np.array(labels)
        return labels

    def _predict_tree(self, x, tree_node):
        # Recursive function to traverse the decision tree
        l = -1
        if tree_node['type'] == 'leaf':
            return tree_node['prediction']
        else:
            nextnode = None
            if x[tree_node['feature_idx']] <= tree_node["threshold"]:
                nextnode = tree_node['left_tree']
            else:
                nextnode = tree_node['right_tree']
            
            l = self._predict_tree(x, nextnode)
        
        return l


    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # split one node into left and right node 
        feature_name = X.columns[feature_index]
        lm = X[feature_name] <= threshold
        rm = X[feature_name] > threshold

        left_dataset_X, left_dataset_y = X[lm], y[lm]
        right_dataset_X, right_dataset_y = X[rm], y[rm]

        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        # Use Information Gain to find the best split for a dataset
        best_gain = 1e-6
        best_feature_index = -1
        best_threshold = -1.0
        e = self._entropy(y)

        #g = len(X.columns)
        #with tqdm(total=g, desc="bsplit", unit="colunm") as pbar:
        for f_idx, f in enumerate(X.columns):
                v = np.sort(X[f].unique())
                if len(v) >= 2:
                    ts = (v[:-1]+v[1:])/2
                    for t in ts:
                        lx, ly, rx, ry = self._split_data(X, y, f, t)

                        if len(ly) != 0 and len(ry) != 0:
                            pl = len(ly)/len(y)
                            pr = 1-pl
                            el = self._entropy(ly)
                            er = self._entropy(ry)

                            g = e - (pl*el+pr*er)

                            if g > best_gain:
                                best_gain = g
                                best_feature_index = f_idx
                                best_threshold = t
                
                #pbar.set_postfix(g=best_gain, i=best_feature_index)
                #pbar.update(1)
        
        return best_feature_index, best_threshold
    
    def _entropy(self, y: np.ndarray)->float:
        # Return the entropy
        v, c = np.unique(y, return_counts=True)
        nv = len(v)
        n = len(y)

        p = c /n
        e = -np.sum(p * np.log2(p + 1e-9))

        return e
    
def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[pd.DataFrame, np.ndarray]:
    # Use the model to extract features from the dataloader, return the features and labels
    model.eval()

    batch = len(dataloader)
    f, labels, features = [], [], []
    layer = dict(model.named_modules())['model.global_pool.flatten']

    def hookf(module, input, output):
        f.append(output.detach())

    with torch.no_grad(), tqdm(total=batch, desc="Features label relation getting Progress", unit="batch") as pbar:
        h = layer.register_forward_hook(hookf)
        for batch_idx, (images, l) in enumerate(dataloader):
                images = images.to(device)
                
                outputs = model(images)

                f1 = f[len(f)-1]

                for i in range(len(images)):
                    features.append(f1[i].squeeze().numpy())
                    labels.append(l[i].item())

                pbar.update(1)
        f = []
        h.remove()
    features = pd.DataFrame(features)
    labels = np.array(labels)
    return features, labels

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[pd.DataFrame, np.ndarray]:
    # Use the model to extract features from the dataloader, return the features and path of the images
    model.eval()

    dataset = dataloader.dataset
    batch = len(dataloader)
    f, paths, features = [], [], []

    layer = dict(model.named_modules())['model.global_pool.flatten']

    def hookf(module, input, output):
        f.append(output.detach())

    with torch.no_grad(), tqdm(total=batch, desc="Features label relation getting Progress", unit="batch") as pbar:
        h = layer.register_forward_hook(hookf)
        for batch_idx, (images, l) in enumerate(dataloader):
                images = images.to(device)
                
                outputs = model(images)
               
                f1 = f[len(f)-1]

                for i in range(len(images)):
                    idx = batch_idx*32 + i
                    path = dataset.image[idx]
                    features.append(f1[i].squeeze().numpy())
                    paths.append(path)

                pbar.update(1)
        f = []
        h.remove()

    features = pd.DataFrame(features)
    paths = np.array(paths)
    return features, paths