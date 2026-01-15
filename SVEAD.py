import torch
import numpy as np

class SVEAD:
    def __init__(self, max_samples=16, n_estimators=100, random_state=None):
        self.max_samples = max_samples
        self.t = n_estimators
        self.random_state = random_state
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def fit(self, X):
        if isinstance(X, np.ndarray):
            self.X_orig = torch.from_numpy(X).to(torch.float32).to(self.device)
        else:
            self.X_orig = X.to(torch.float32).to(self.device)
            
        n_samples = self.X_orig.shape[0]
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        indices = [np.random.choice(n_samples, min(self.max_samples, n_samples), replace=False) for _ in range(self.t)]
        self.samples_tensor = torch.stack([self.X_orig[i] for i in indices]) # [T, M, D]

        self.global_max_dist = torch.zeros((self.t, self.max_samples), dtype=torch.float32, device=self.device)
        self.global_sum_dist = torch.zeros((self.t, self.max_samples), dtype=torch.float32, device=self.device)
        self.global_count = torch.zeros((self.t, self.max_samples), dtype=torch.float32, device=self.device)

        fit_batch_size = 100000 
        for i in range(0, n_samples, fit_batch_size):
            end = min(i + fit_batch_size, n_samples)
            batch_x = self.X_orig[i:end].unsqueeze(0) # [1, batch_n, D]
            
            # 计算距离（float32）
            dists = torch.cdist(batch_x, self.samples_tensor, p=2) # [T, batch_n, M]
            min_d, min_idx = torch.min(dists, dim=2) # [T, batch_n]

            self.global_max_dist.scatter_reduce_(1, min_idx, min_d, reduce='amax', include_self=True)
            self.global_sum_dist.scatter_add_(1, min_idx, min_d)
            self.global_count.scatter_add_(1, min_idx, torch.ones_like(min_d))

        self.global_mean_dist = self.global_sum_dist / (self.global_count + 1e-9)
        self.global_max_dist = torch.clamp(self.global_max_dist, min=1e-9)


        return self
        
    def decision_function(self, X=None, batch_size=100000):
        # 确保输入是 float32
        if X is None:
            X_tensor = self.X_orig
        else:
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        
        N = X_tensor.shape[0]
        all_scores = np.zeros(N, dtype=np.float32)
        
        for start_it in range(0, N, batch_size):
            end_it = min(start_it + batch_size, N)
            X_batch = X_tensor[start_it:end_it].unsqueeze(0)
            
            dists = torch.cdist(X_batch, self.samples_tensor, p=2) 
            nearest_dist, nearest_idx = torch.min(dists, dim=2) 

            sample_max_dist = torch.gather(self.global_max_dist, 1, nearest_idx)
            sample_mean_dist = torch.gather(self.global_mean_dist, 1, nearest_idx)
            
            batch_scores = (nearest_dist / sample_max_dist) * sample_mean_dist

            batch_scores = torch.nan_to_num(batch_scores, nan=0.0)
            
            all_scores[start_it:end_it] = torch.mean(batch_scores, dim=0).cpu().numpy()

        return all_scores
