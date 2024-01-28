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

    return acc_value, max_predicted_label

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
    def __init__(self, training_arguments, model_arguments, other_arguments):
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
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

        self.optimizer = Adam
        self.save_hyperparameters("training_arguments")
        self.save_hyperparameters("model_arguments")

        self.loss_values_list = []

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
        acc, predicted_label = compute_accuracy(student_distribution, y)
        self.log('train_loss_inbal', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        # print probabilities
        self.loss_values_list += [loss, acc, self.trainer.current_epoch, self.trainer.global_step]
        parameters = [loss, acc, self.trainer.current_epoch, self.trainer.global_step]

        file = open("loss_values.txt", "a")
        for parameter in parameters:
            file.write('%s,' % parameter)
        file.write('\n')
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

    def setup(self, stage=None):
        dataset = load_dataset("csv", data_files="final_dataset_2_example.csv")

        number_of_train_samples = len(dataset)
        if self.other_arguments.max_train_samples != -1:
            number_of_train_samples = min(self.other_arguments.max_train_samples, number_of_train_samples)

    def train_dataloader(self):
        dataset = load_dataset("csv", data_files="final_dataset_2_example.csv")

        dataset = dataset["train"]
        dataloader = DataLoader(dataset, self.other_arguments.train_batch_size, drop_last=False,
                                    shuffle=True, num_workers=self.training_arguments.num_workers)

        return dataloader

    def calculate_student_model_distribution(self, input, y):
        vocabulary = self.tokenizer.get_vocab()
        labels = list(vocabulary.keys())
        class_ids = torch.LongTensor(self.tokenizer(labels, padding="longest").input_ids)

        teacher_distribution_tensor = create_y_tensor(y)
        teacher_label_index = teacher_distribution_tensor.max(dim=1)[1][0]
        word = labels[teacher_label_index]

        encoding = self.tokenizer(input, return_tensors="pt", return_length=True)
        labels = self.tokenizer(input, return_tensors="pt").input_ids
        # labels = self.tokenizer(word, return_tensors="pt").input_ids

        model.train()
        generated_outputs = self.model(input_ids=encoding.input_ids, labels=labels)

        score_of_labels = generated_outputs.logits.gather(dim=2, index=class_ids.T.expand(1, -1, -1))

        probabilities = score_of_labels.softmax(2)
        return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training arguments
    training_arguments = parser.add_argument_group('training_arguments')
    training_arguments.add_argument("--opt_level", default="O1")
    training_arguments.add_argument('--max_grad_norm', type=float, default=1.0)
    training_arguments.add_argument("--fp_16", default=False, action="store_true")
    training_arguments.add_argument("--n_gpu", default=-1, type=int)
    training_arguments.add_argument("--num_workers", default=8, type=int)
    training_arguments.add_argument("--distributed_backend", default=None)

    # Model arguments
    model_arguments = parser.add_argument_group('model_arguments')
    model_arguments.add_argument("--num_labels", type=int)
    model_arguments.add_argument("--fc1_size", type=int)

    # Other arguments
    other_arguments = parser.add_argument_group('other_arguments')
    other_arguments.add_argument("--output_dir", default="./")
    other_arguments.add_argument("--teacher_model", default="./")
    other_arguments.add_argument("--alpha_for_kd", default=0.9, type=float)
    other_arguments.add_argument("--temperature_for_kd", default=20, type=int)
    other_arguments.add_argument("--predictions_file", default="predictions.csv")
    other_arguments.add_argument("--data_dir", default="./")
    other_arguments.add_argument("--train_batch_size", default=2, type=int)
    other_arguments.add_argument("--eval_batch_size", default=2, type=int)
    other_arguments.add_argument("--max_train_samples", default=-1, type=int)
    other_arguments.add_argument("--num_train_epochs", default=50, type=int)
    other_arguments.add_argument("--gradient_accumulation_steps", default=1, type=int)
    other_arguments.add_argument("--seed", default=42, type=int)
    other_arguments.add_argument("--save_top_k", default=-1, type=int)
    other_arguments.add_argument("--save_last", default=False, action="store_true")
    other_arguments.add_argument("--write_dev_predictions", default=False, action="store_true")
    # other_arguments.add_argument('--learning_rate', type=float, default=0.03)
    other_arguments.add_argument('--learning_rate', type=float, default=3e-4)

    other_arguments.add_argument("--do_fast_dev_run", default=False, action="store_true")
    other_arguments.add_argument("--limit_train_batches", default=-1, type=int)
    other_arguments.add_argument("--limit_val_batches", default=-1, type=int)

    '''
    args = parser.parse_args(
    " --model_name_or_path roberta-base  --max_input_seq_length 100   --TRAIN_FILE sst2_train.csv --output_dir ./ --DEV_FILE sst2_dev.csv --train_batch_size 32 --eval_batch_size 32 --max_train_samples 10000 --num_train_epochs 5 --gradient_accumulation_steps 1 --save_top_k 0 --learning_rate 5e-6 --write_dev_predictions".split()
    )
    '''
    args = parser.parse_args()

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if (group.title == "training_arguments"):
            training_arguments = argparse.Namespace(**group_dict)
        elif (group.title == "model_arguments"):
            model_arguments = argparse.Namespace(**group_dict)
        elif (group.title == "other_arguments"):
            other_arguments = argparse.Namespace(**group_dict)

    print("Training arguments", training_arguments)
    print("--------------------")
    print("Model arguments", model_arguments)
    print("--------------------")
    print("Other arguments", other_arguments)
    print("--------------------")

    pl.seed_everything(other_arguments.seed)

    # teacher_model = ClassificationModel.load_from_checkpoint(checkpoint_path=other_arguments.teacher_model,
    #                                                          other_arguments=None)
    model = ClassificationModelKD(training_arguments=training_arguments,
                                model_arguments=model_arguments,
                                other_arguments=other_arguments)
                                # teacher_model=teacher_model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=other_arguments.output_dir,
        monitor="val_acc",
        save_top_k=other_arguments.save_top_k,
        save_last=other_arguments.save_last,
        mode='max'
    )

    train_params = dict(
        # accumulate_grad_batches=other_arguments.gradient_accumulation_steps,
        # gpus=training_arguments.n_gpu,
        # deterministic=True,
        max_epochs=other_arguments.num_train_epochs,
        precision=16 if training_arguments.fp_16 else 32,
        # amp_level=training_arguments.opt_level,
        # gradient_clip_val=training_arguments.max_grad_norm,
        # callbacks=[checkpoint_callback],
        # fast_dev_run=other_arguments.do_fast_dev_run,
    )

    if (other_arguments.limit_train_batches != -1):
        train_params["limit_train_batches"] = other_arguments.limit_train_batches

    if (other_arguments.limit_val_batches != -1):
        train_params["limit_val_batches"] = other_arguments.limit_val_batches

    if (training_arguments.distributed_backend != None):
        train_params["distributed_backend"] = training_arguments.distributed_backend

    file = open("loss_values.txt", "w")
    file.writelines("loss,accuracy,epoch,step")
    file.close()

    file1 = open("word_10747_student_probability.txt", "w")
    file1.writelines("probability")
    file1.close()

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


