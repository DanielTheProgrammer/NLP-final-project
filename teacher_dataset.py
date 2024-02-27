import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast


class TeacherDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data.iloc[idx, :-1].values.tolist()
        # label = self.data.iloc[idx, -1]
        label_str = self.data.iloc[idx, -1]
        label = ast.literal_eval(label_str)  # Parse string to list
        return input_data, label


def create_dataloader(csv_file_path, batch_size):
    dataset = TeacherDataset(csv_file_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
