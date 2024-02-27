import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import ast


class TeacherDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # input_data = self.data.iloc[idx, :-1].values.tolist()
        # # label = self.data.iloc[idx, -1]
        # label_str = self.data.iloc[idx, -1]
        # label = ast.literal_eval(label_str)  # Parse string to list
        # return input_data, label

        # Get input and label data at the specified index
        input_text = self.data.iloc[idx, 0]  # Assuming input is in the first column
        label_str = self.data.iloc[idx, 1]  # Assuming label is in the second column

        # Parse label string to extract indices and probabilities
        label_list = eval(label_str)  # Convert string representation to list

        # Convert label list to tensors
        indices = torch.tensor([int(item[0]) for item in label_list], dtype=torch.long)
        probabilities = torch.tensor([float(item[1]) for item in label_list], dtype=torch.float)

        # Return input and label as tensors
        return input_text, (indices, probabilities)


def create_dataloader(csv_file_path, batch_size):
    dataset = TeacherDataset(csv_file_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
