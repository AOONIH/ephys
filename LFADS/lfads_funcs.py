import joblib
import numpy as np
import torch
from lfads_torch.model import LFADS
from lfads_torch.trainers import LFADSTrainer
import matplotlib.pyplot as plt


class LFADSRunner:
    def __init__(
        self,
        input_size,
        factors_size=10,
        latent_size=20,
        encoder_size=64,
        controller_size=64,
        co_dim=0,
        dropout=0.05,
        kl_weight=1.0,
        l2_gen_scale=1e-4,
        l2_con_scale=1e-4,
        device='cuda',
        save_dir='lfads_output',
        max_epochs=100,
        batch_size=64,
        kl_anneal=False,
    ):
        self.device = device
        self.save_dir = save_dir

        # Initialize LFADS model
        self.model = LFADS(
            input_size=input_size,
            factors_size=factors_size,
            latent_size=latent_size,
            encoder_size=encoder_size,
            controller_size=controller_size,
            co_dim=co_dim,
            dropout=dropout,
            kl_weight=kl_weight,
            l2_gen_scale=l2_gen_scale,
            l2_con_scale=l2_con_scale,
            device=device,
        )

        # Initialize trainer
        self.trainer = LFADSTrainer(
            model=self.model,
            save_dir=self.save_dir,
            max_epochs=max_epochs,
            batch_size=batch_size,
            kl_anneal=kl_anneal,
        )

    def fit(self, data):
        """
        Train LFADS model on data.
        Args:
            data (torch.Tensor): shape (trials, units, time)
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        self.trainer.fit(data)

    def get_factors(self, data):
        """
        Extract latent factors from trained LFADS model.
        Returns:
            torch.Tensor: shape (trials, factors, time)
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        return self.model.get_factors(data)

    def reconstruct(self, data):
        """
        Return LFADS-reconstructed firing rates.
        Returns:
            torch.Tensor: shape (trials, units, time)
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        return self.model.reconstruct(data)

def plot_reconstruction(original, recon, trial_idx=0, unit_idx=0):

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(original[trial_idx, unit_idx], label='Original')
    ax.plot(recon[trial_idx, unit_idx], label='Reconstructed', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Firing Rate')
    ax.set_title(f'Trial {trial_idx}, Unit {unit_idx}')
    ax.legend()
    fig.tight_layout()
    fig.show()


def plot_latent_trajectories(factors, trial_idx=0, method='pca'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    import numpy as np

    traj = factors[trial_idx].T  # (time, factors)

    if method == 'pca':
        reduced = PCA(n_components=3).fit_transform(traj)
    elif method == 'umap':
        import umap
        reduced = umap.UMAP(n_components=3).fit_transform(traj)
    else:
        raise ValueError("Use 'pca' or 'umap'")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*reduced.T)
    ax.set_title(f'Latent Trajectory (Trial {trial_idx})')
    fig.tight_layout()
    fig.show()


def plot_r2_histogram(original, recon):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    import numpy as np

    trials, units, time = original.shape
    r2s = []
    for unit in range(units):
        y_true = original[:, unit, :].reshape(-1)
        y_pred = recon[:, unit, :].reshape(-1)
        r2s.append(r2_score(y_true, y_pred))

    r2s = np.array(r2s)
    print("Median R²:", np.median(r2s))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2s, bins=30)
    ax.set_title("R² Across Units")
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.show()

    return r2s



if __name__ == '__main__':
    # Assume `data` is a (trials, units, time) numpy array
    data_dict = joblib.load(r"/ceph/akrami/Dammy/aggr_savedir/normdev_resps_ephys_2401_2504.joblib")
    data = np.stack(list(data_dict.values()),axis=0)

    lfads = LFADSRunner(input_size=data.shape[1], max_epochs=50)

    # Train model
    lfads.fit(data)

    # Get latent factors
    factors = lfads.get_factors(data)  # (trials, factors, time)

    # Get denoised reconstructions
    recon = lfads.reconstruct(data)