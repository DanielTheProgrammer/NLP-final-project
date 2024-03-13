import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.optim import Adam
from torchvision import transforms
from transformers import T5Tokenizer


class ClassificationModelKD(pl.LightningModule):
    def __init__(self, training_arguments, model_arguments, other_arguments, student_model, dataloader):
        super(ClassificationModelKD, self).__init__()

        self.training_arguments = training_arguments
        self.model_arguments = model_arguments
        self.other_arguments = other_arguments

        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = student_model

        self.optimizer = Adam
        self.save_hyperparameters("training_arguments")
        self.save_hyperparameters("model_arguments")

        self.loss_values_list = []
        self.dataloader = dataloader

        file = open("model_run_parameters.txt", "w")
        file.writelines("epoch,step,input,loss,accuracy,student_probability_for_true_label,true_probability_of_label\n")

        file = open("student_model_generated.txt", "w")
        file.writelines("epoch,input,output_1,output_2,output_3\n")
        file.close()

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(self, x):
        x = self.model(x)
        return x

    def _step(self, batch):
        x, y = batch

        alpha = self.other_arguments.alpha_for_kd
        T = self.other_arguments.temperature_for_kd

        y_tensor = self.create_y_tensor(y)

        student_distribution = self.calculate_student_model_distribution(x, y_tensor)
        loss = self.compute_loss(student_distribution, y_tensor, T, alpha)

        return loss, student_distribution

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, student_distribution = self._step(batch)

        y_tensor = self.create_y_tensor(y)
        acc, predicted_label, student_probability_of_label, true_probability_of_label =\
            self.compute_accuracy(student_distribution, y_tensor)

        self.log('train_loss_inbal', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        return {"loss": loss, "acc": acc}


    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.other_arguments.learning_rate)

    def train_dataloader(self):
        return self.dataloader

    def calculate_student_model_distribution(self, inputs, y_tensor):
        vocabulary = self.tokenizer.get_vocab()
        vocabulary_list = list(vocabulary.keys())
        class_ids = torch.LongTensor(self.tokenizer(vocabulary_list, padding="longest").input_ids)

        batch_size = len(y_tensor)  # in case the batch size is larger than the remaining items on the list
        encoding = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", return_length=True)
        input_labels = [inputs[i] + self.get_correct_label(y_tensor[i]) for i in range(batch_size)]
        labels = self.tokenizer(input_labels, padding=True, truncation=True, return_tensors="pt", return_length=True)

        generated_outputs = self.model(input_ids=encoding.input_ids, labels=labels.input_ids)

        probabilities = F.softmax(generated_outputs.logits, dim=-1)
        return probabilities

    def get_correct_label(self, teacher_distribution_tensor):
        teacher_label_index = teacher_distribution_tensor.max(dim=0).indices.item()
        label = self.tokenizer.convert_ids_to_tokens(teacher_label_index)
        return label

    def compute_accuracy(self, student_distribution, teacher_distribution):
        acc_value = 0
        batch_size = len(teacher_distribution)
        max_predicted_labels = torch.zeros(batch_size)
        student_distribution_for_max_label = torch.zeros(batch_size)
        teacher_distribution_for_max_label = torch.zeros(batch_size)
        for i in range(batch_size):
            max_predicted_probability, max_predicted_label = torch.max(student_distribution[i][0], dim=0)
            teacher_label_index = teacher_distribution[i].max(dim=0).indices.item()

            max_predicted_labels[i] = max_predicted_label
            student_distribution_for_max_label[i] = student_distribution[i][0][teacher_label_index].item()
            teacher_distribution_for_max_label[i] = teacher_distribution[i][teacher_label_index].item()

            identical_max_labels = (max_predicted_label == teacher_label_index)
            if identical_max_labels:
                acc_value += 1
            else:
                acc_value += 0

        acc_value /= batch_size
        return acc_value, max_predicted_labels, student_distribution_for_max_label, teacher_distribution_for_max_label

    def create_y_tensor(self, y):
        batch_size = len(y[0])
        y_tensor = torch.zeros(batch_size, 32128)

        for i in range(batch_size):
            for j in range(len(y[0][i])):
                y_tensor[i][y[0][i][j]] = y[1][i][j]
        return y_tensor

    def compute_loss(self, student_distribution, teacher_distribution, T, alpha):
        filtered_student_distribution = student_distribution[0][0].reshape(1, -1)
        custom_loss = torch.nn.KLDivLoss()(F.log_softmax(filtered_student_distribution / T, dim=1),
                                           F.softmax(teacher_distribution / T, dim=1)) * (alpha * T * T)
        return custom_loss

    def on_train_epoch_end(self):
        super(ClassificationModelKD, self).on_train_epoch_end()

