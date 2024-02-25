import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, TFT5Model
from datasets import load_dataset, concatenate_datasets
import ast


def compute_accuracy(student_distribution, labels):
    max_predicted_probability, max_predicted_label = torch.max(student_distribution[0][0], dim=0)

    teacher_distribution_tensor = create_y_tensor(labels)
    teacher_label_index = teacher_distribution_tensor.max(dim=1)[1][0]

    acc = (max_predicted_label == teacher_label_index)
    if acc:
        acc_value = 1
    else:
        acc_value = 0
    file = open("word_10747_student_probability.txt", "a")
    file.write('%f' % student_distribution[0][0][7163])
    file.write('\n')
    file.close()

    return acc_value, max_predicted_label, student_distribution[0][0][teacher_label_index], \
           teacher_distribution_tensor[0][teacher_label_index]

def create_y_tensor(y):
    y_arr = ast.literal_eval(y)
    y_tensor = torch.zeros(1, 32100)
    for idx, prob in y_arr:
        idx = int(idx)
        y_tensor[0][idx] = prob
    return y_tensor


def compute_loss(student_distribution, y, T, alpha):
    filtered_student_distribution = student_distribution[0][0].reshape(1, -1)
    teacher_distribution = create_y_tensor(y)

    custom_loss = torch.nn.KLDivLoss()(F.log_softmax(filtered_student_distribution / T, dim=1),
                                       F.softmax(teacher_distribution / T, dim=1)) * (alpha * T * T)
    return custom_loss


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
        # self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

        self.optimizer = Adam
        self.save_hyperparameters("training_arguments")
        self.save_hyperparameters("model_arguments")

        self.loss_values_list = []
        self.dataloader = dataloader

        file = open("model_run_parameters.txt", "w")
        file.writelines("epoch,step,loss,accuracy,student_probability_for_true_label,true_probability_of_label\n")
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

        student_distribution = self.calculate_student_model_distribution(x, y)
        loss = compute_loss(student_distribution, y, T, alpha)

        return loss, student_distribution


    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, student_distribution = self._step(batch)
        acc, predicted_label, student_probability_of_label, true_probability_of_label =\
            compute_accuracy(student_distribution, y)
        self.log('train_loss_inbal', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        # print probabilities
        self.loss_values_list += [loss, acc, self.trainer.current_epoch, self.trainer.global_step]
        parameters = [self.trainer.global_step, self.trainer.current_epoch, loss, acc, student_probability_of_label,
                      true_probability_of_label]

        file = open("model_run_parameters.txt", "a")
        line = str(self.trainer.global_step) + "," + str(self.trainer.current_epoch) + "," + str('%.8f' % loss.item()) + "," + \
                str('%.8f' % acc) + "," + str('%.8f' % student_probability_of_label.item()) + "," + str('%.8f' % true_probability_of_label.item()) + '\n'
        file.write(line)
        file.close()
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, student_distribution = self._step(batch)
        student_distribution = student_distribution.squeeze(1)
        acc, predicted_label = compute_accuracy(student_distribution, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return {
            "val_loss": loss,
            "val_acc": acc,
            "softmax_logits": student_distribution.tolist(),
            "labels": y.tolist(),
            "predictions": predicted_label.tolist(),
        }

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'].view(-1) for x in outputs]).mean()
        avg_acc = torch.cat([x['val_acc'].view(-1) for x in outputs]).mean()

        all_labels = []
        all_predictions = []
        all_softmax_logits = []

        for x in outputs:
            all_predictions += torch.tensor(x["predictions"]).tolist()
            all_softmax_logits += torch.tensor(x["softmax_logits"]).tolist()
            all_labels += torch.tensor(x["labels"]).tolist()

        softmax_logits_df = pd.DataFrame(all_softmax_logits)
        print("--------------------")
        print("Validation avg_loss: ", avg_loss)
        print("Validation avg_acc: ", avg_acc)

        result_df = pd.DataFrame({
            "label": all_labels,
            "prediction": all_predictions,
        })

        result_df = pd.concat([result_df, softmax_logits_df], axis=1)

        if (self.other_arguments.write_dev_predictions):
            output_path = self.other_arguments.output_dir + "epoch_" + str(
                self.trainer.current_epoch) + "_" + self.other_arguments.predictions_file
            print(f"Writing predictions for dev to {output_path}")
            result_df.to_csv(output_path, index=False)
        print("--------------------")

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.other_arguments.learning_rate)

    def train_dataloader(self):
        return self.dataloader

    def calculate_student_model_distribution(self, input, y):
        vocabulary = self.tokenizer.get_vocab()
        labels = list(vocabulary.keys())
        class_ids = torch.LongTensor(self.tokenizer(labels, padding="longest").input_ids)

        # teacher_distribution_tensor = create_y_tensor(y)
        # teacher_label_index = teacher_distribution_tensor.max(dim=1)[1][0]
        # word = labels[teacher_label_index]

        encoding = self.tokenizer(input, return_tensors="pt", return_length=True)
        labels = self.tokenizer(input, return_tensors="pt").input_ids
        # labels = self.tokenizer(word, return_tensors="pt").input_ids

        # self.model.train()
        generated_outputs = self.model(input_ids=encoding.input_ids, labels=labels)
        # generated_outputs = self.model(input)

        score_of_labels = generated_outputs.logits.gather(dim=2, index=class_ids.T.expand(1, -1, -1))

        probabilities = score_of_labels.softmax(2)
        return probabilities

    def on_train_epoch_end(self):
        file = open("student_model_generated.txt", "a")
        for x, label in self.dataloader:
            input_ids = self.tokenizer(x, return_tensors="pt").input_ids
            generated_output = self.model.generate(input_ids, num_return_sequences=3, output_scores=True,
                                                   return_dict_in_generate=True, num_beams=3)
            generated_sequences = generated_output.sequences
            decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in generated_sequences]

            line = str(self.trainer.current_epoch) + "," + x + ","
            file.write(line)

            for output in decoded_outputs:
                file.write(f',{output}')
            file.write('\n')
        file.close()

        super(ClassificationModelKD, self).on_train_epoch_end()

