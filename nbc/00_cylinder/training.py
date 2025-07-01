import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from lbm import *
import h5py
from typing import Optional
from torch.utils import data

class HDF5Dataset(data.Dataset):
    """ Custom dataset for HDF5 files that can be used by torch's dataloader.

    Parameters
    ----------
        filebase : string
            Path to the hdf5 file with annotations.
        transform : class object
            Optional transform to be applied on a f loaded from HDF5 file.
        target : logical operation (True, False)
            Returns also the next dataset[idx + skip_idx_to_target] - default=False
        skip_idx_to_target : integer
            Define which next target dataset is returned if target is True - default=1

    Examples
        --------
        Create a data loader.
        >>> import torch
        >>> dataset_train = HDF5Dataset(filebase= "./hdf5_output.h5",
        >>>                             target=True)
        >>> train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True)
        >>> for (f, target, idx) in train_loader:
        >>>     ...
        """

    def __init__(self, filebase,
                 transform=None,
                 target: bool = False,
                 skip_idx_to_target: int = 1,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = "cpu"):
        super().__init__()
        self.filebase = filebase
        self.transform = transform
        self.target = target
        self.skip_idx_to_target = skip_idx_to_target
        self.fs = h5py.File(self.filebase, "r")
        self.shape = self.fs["f"].shape
        self.keys = list(self.fs.keys())
        # ToDo: the following line is lettuce specific. Change it to general dataset
        self.dtype = dtype
        self.device = device

    def __str__(self):
        for attr, value in self.fs.attrs.items():
            # ToDo: the following 2 lines are lettuce specific. Change it to general dataset
            if attr in ('flow', '_collision'):
                print(attr + ": " + str(self._unpickle_from_h5(self.fs.attrs[attr])))
            else:
                print(attr + ": " + str(value))
        return ""

    def __len__(self):
        return self.shape[0] - self.skip_idx_to_target if self.target else self.shape[0]

    def __getitem__(self, idx):
        f = self.get_data(idx)
        target = []
        if self.target:
            target = self.get_data(idx + self.skip_idx_to_target)
        if self.transform:
            f = self.transform(f)
            if self.target:
                target = self.transform(target)
        return (f, target, idx) if self.target else (f, idx)

    def __del__(self):
        self.fs.close()

    def get_data(self, idx):
        return self._convert_to_tensor(self.fs["f"][idx])

    def get_attr(self, attr):
        return self.fs.attrs[attr]

    @staticmethod
    def _unpickle_from_h5(byte_str):
        return pickle.load(io.BytesIO(byte_str))


    def _convert_to_tensor(self, array, *args,
                          dtype: Optional[torch.dtype] = None, **kwargs
                          ) -> torch.Tensor:
        is_tensor = isinstance(array, torch.Tensor)
        new_dtype = dtype
        if dtype is None:
            if hasattr(array, 'dtype'):
                if array.dtype in [bool, torch.bool]:
                    new_dtype = torch.bool
                elif array.dtype in [bool, torch.uint8, np.uint8]:
                    new_dtype = torch.uint8
                else:
                    new_dtype = self.dtype
            else:
                new_dtype = self.dtype

        if is_tensor:
            return array.to(*args, **kwargs, device=self.device,
                            dtype=new_dtype)
        else:
            return torch.tensor(array, *args, **kwargs, device=self.device,
                                dtype=new_dtype)

class ShiftedSigmoid(torch.nn.Module):
    def forward(self, x):
        """Apply sigmoid transformation with a shift."""
        return torch.sigmoid(x) - 0.5

class NeuralBoundary(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device='cpu', nodes=20, index=None):
        """Initialize a neural network boundary model."""
        super(NeuralBoundary, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, nodes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, nodes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, 3, bias=True)
        ).to(dtype=dtype, device=device)
        self.index = index

    def forward(self, x):
        """Forward pass through the network with residual connection."""
        return self.net(x) + x[:,self.index]

def plot_velocity_field(u, title, ax, vmin=None, vmax=None):
    """Helper function to plot the velocity field."""
    ax.imshow(u, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')  # Hide axes

def compute_reference_velocity(f, d2q9):
    """Compute the initial reference velocity field for comparison."""
    return np.linalg.norm(velocity(f.cpu().numpy(), d2q9), axis=0)

def train_model(model, optimizer, criterion, f, x_range, num_epochs=1, index=None):
    """Train the model and return epoch losses."""
    sequence = torch.randperm(x_range[1] - x_range[0]-1)
    epoch_loss = []

    for _ in tqdm(range(num_epochs), desc="Training Progress"):
        running_loss = 0.0
        for idx in sequence:
            optimizer.zero_grad()
            inputs = f[:, idx, :].T
            reference = f[index, idx + 1, :].T
            outputs = model(inputs)
            loss = criterion(outputs, reference)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss.append(running_loss)
    return epoch_loss

def expand_velocity_field(model, f, d2q9, expand_steps, index=[3,6,7]):
    """Expand the velocity field based on the trained model."""
    f_expanded = f.clone()
    for _ in range(expand_steps):
        f_new = f_expanded[:,-1,:].unsqueeze(1)
        f_new[index,...] = model(f_new[:, 0, :].T).T.unsqueeze(1)
        f_expanded = torch.cat((f_expanded, f_new), dim=1)
    u_expanded = np.linalg.norm(velocity(f_expanded.cpu().numpy(), d2q9), axis=0)
    return u_expanded

def plot_training_results(epoch_loss, u, u_reference, x1_x0, vmin, vmax):
    """Plot the training loss and the comparison of velocity fields."""
    plt.figure()
    plt.plot(epoch_loss)
    plt.yscale('log')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    u[x1_x0-1, :] = 1  # Highlight boundary
    plot_velocity_field(u, "NBC", ax[0], vmin, vmax)

    u_reference[x1_x0-1, :] = 1  # Highlight boundary in reference
    plot_velocity_field(u_reference, "Original", ax[1], vmin, vmax)
    plt.show()

def main(filebase="./hdf5_output.h5"):
    torch.manual_seed(1)
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    index = [3,6,7]
    # Initialize training
    dataset_train = HDF5Dataset(filebase=filebase, target=False)
    model = NeuralBoundary(dtype, device, nodes=20, index=index)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    d2q9 = D2Q9()

    # Load data
    f_original = dataset_train[0][0].to(dtype=dtype, device=device)
    del dataset_train
    #
    x_range = (0, 1500)
    # y_range = (0, -1)
    u_original = compute_reference_velocity(f_original, d2q9)
    plot_velocity_field(u_original, "Original", plt.gca())

    num_epochs = 200
    epoch_loss = train_model(model, optimizer, criterion, f_original, x_range, num_epochs, index)
    # model = torch.load("model_training_v1.pt", weights_only=False)
    # epoch_loss = 0
    model.eval()
    torch.save(model, "model_trained.pt")
    print("Training and evaluation completed.")

if __name__ == "__main__":
    main(filebase="./cylinder_BGK_Re200_GPD50_DpX100_DpY50_mach0.1_training.h5")