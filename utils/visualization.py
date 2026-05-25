import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


class TensorVisualizer:
    def __init__(self, tensor_list):
        if isinstance(tensor_list, torch.Tensor):
            self.data = tensor_list.detach().cpu().numpy()
        elif isinstance(tensor_list, list):
            if isinstance(tensor_list[0], torch.Tensor):
                self.data = np.stack([t.detach().cpu().numpy() for t in tensor_list])
            else:
                self.data = np.stack(tensor_list)
        else:
            self.data = tensor_list
        self.L, self.B, self.M, self.C = self.data.shape

    def _ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def plot_heatmaps(self, batch_idx=0, cols=3, cmap='viridis', save_path=None):
        rows = (self.L + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten() if self.L > 1 else [axes]
        fig.suptitle(f'Heatmaps for Batch Index: {batch_idx} (Shape: M x C)', fontsize=16)
        for i in range(self.L):
            matrix = self.data[i, batch_idx, :, :]
            im = axes[i].imshow(matrix, aspect='auto', cmap=cmap)
            axes[i].set_title(f'Layer {i}')
            axes[i].set_xlabel('Channels (C)')
            axes[i].set_ylabel('Seq/Items (M)')
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_pca_evolution(self, batch_idx=0, save_path=None):
        target_data = self.data[:, batch_idx, :, :]
        reshaped_data = target_data.reshape(-1, self.C)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(reshaped_data)
        fig = plt.figure(figsize=(10, 8))
        colors = plt.cm.jet(np.linspace(0, 1, self.L))
        for i in range(self.L):
            start_idx = i * self.M
            end_idx = (i + 1) * self.M
            layer_points = reduced_data[start_idx:end_idx]
            plt.scatter(layer_points[:, 0], layer_points[:, 1],
                        color=colors[i], label=f'Layer {i}', alpha=0.6, s=50)
            if i > 0 and self.M < 20:
                prev_layer_points = reduced_data[start_idx - self.M: end_idx - self.M]
                for m in range(self.M):
                    plt.plot([prev_layer_points[m, 0], layer_points[m, 0]],
                             [prev_layer_points[m, 1], layer_points[m, 1]],
                             color='gray', alpha=0.3, linewidth=0.5)
        plt.title(f'PCA Projection (Batch {batch_idx})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            self._ensure_dir(save_path)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
