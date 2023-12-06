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
import Distilation_Trainer


def compute_accuracy(logits, labels):
    # predicted_label = logits.max(dim=1)[1]
    predicted_label = logits.max(dim=1)[1][0]
    newTorch = torch.zeros(1, )
    newTorch[0] = predicted_label[0]

    y_tensor = create_y_tensor(labels)
    teacher_label = y_tensor.max(dim=1)[1]

    # acc = (predicted_label == teacher_label)
    acc = (newTorch == teacher_label)
    if acc[0]:
        accValue = 1
    else:
        accValue = 0
    return accValue, predicted_label

def create_y_tensor(y):
    y_arr = ast.literal_eval(y)
    y_tensor = torch.zeros(1, 32100)
    for idx, prob in y_arr:
        idx = int(idx)
        y_tensor[0][idx] = prob
    return y_tensor


def compute_loss(student_distribution, y, T, alpha):
    # calculate the T5 model probabilities over the input
    # T5_probabilities = model.calculate_T5_probabilities(inputs)
    # logits = student_distribution.logits

    filtered_student_distribution = student_distribution[0][0:3]
    teacher_distribution = create_y_tensor(y)

    # custom_loss = torch.nn.KLDivLoss()
    custom_loss = torch.nn.KLDivLoss()(F.log_softmax(filtered_student_distribution / T, dim=1),
                                       F.softmax(teacher_distribution / T, dim=1)) * (alpha * T * T)
    custom_loss.requires_grad = True
    # custom_loss = torch.nn.KLDivLoss()(F.log_softmax(student_distribution / T, dim=1),
    #                                    F.softmax(teacher_distribution / T, dim=1)) * (alpha * T * T) + \
    #        F.nll_loss(student_distribution, teacher_distribution) * (1. - alpha)
    # custom_loss = ...
    return custom_loss

# class ClassificationModel(pl.LightningModule):
#     def __init__(self, training_arguments, model_arguments, other_arguments):
#         super(ClassificationModel, self).__init__()
#
#         self.training_arguments = training_arguments
#         self.model_arguments = model_arguments
#         self.other_arguments = other_arguments
#
#         self.dims = (1, 28, 28)
#         channels, width, height = self.dims
#         self.transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,)),
#             ]
#         )
#         self.model = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(channels * width * height, self.model_arguments.fc1_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(self.model_arguments.fc1_size, self.model_arguments.fc1_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(self.model_arguments.fc1_size, self.model_arguments.num_labels),
#         )
#
#         self.optimizer = Adam
#         self.save_hyperparameters("training_arguments")
#         self.save_hyperparameters("model_arguments")
#
#     def is_logger(self):
#         return self.trainer.proc_rank <= 0
#
#     def forward(self, x):
#         x = self.model(x)
#         # x = F.log_softmax(x, dim=1)
#         return x
#
#     def _step(self, batch):
#         x, y = batch
#         outputs = self.model(x)
#         logits = F.log_softmax(outputs, dim=1)
#         softmax_logits = F.softmax(outputs, dim=1)
#         loss = F.nll_loss(logits, y)
#         return loss, softmax_logits


class ClassificationModelKD(pl.LightningModule):
    def __init__(self, training_arguments, model_arguments, other_arguments):
        super(ClassificationModelKD, self).__init__()

        self.training_arguments = training_arguments
        self.model_arguments = model_arguments
        self.other_arguments = other_arguments
        # self.teacher_model = teacher_model

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

        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(channels * width * height, self.model_arguments.fc1_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.model_arguments.fc1_size, self.model_arguments.fc1_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.model_arguments.fc1_size, self.model_arguments.num_labels),
        # )

        self.optimizer = Adam
        self.save_hyperparameters("training_arguments")
        self.save_hyperparameters("model_arguments")

        self.loss_values_list = []

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(self, x):
        x = self.model(x)
        # x = F.log_softmax(x, dim=1)
        return x

    def _step(self, batch):
        x, y = batch
        # with torch.no_grad():
        #     output_teacher_batch = self.teacher_model(x)
        # y = self.create_y_tensor(y)

        alpha = self.other_arguments.alpha_for_kd
        T = self.other_arguments.temperature_for_kd

        # encoded_input = self.tokenizer(x, return_tensors='pt')

        # input_ids = self.tokenizer(x, return_tensors="pt").input_ids
        # decoder_input_ids = self.tokenizer(y, return_tensors="pt").input_ids
        # decoder_input_ids = self.model._shift_right(decoder_input_ids)
        # add labels to the inputs, maybe decoder_output_ids

        # input_ids = self.tokenizer.encode(x, return_tensors="pt")
        # input_ids = input_ids.type(torch.LongTensor)
        # input_ids = input_ids.to_sparse()
        # labels = self.tokenizer.encode(y, return_tensors="pt")
        # labels = y.type(torch.LongTensor)
        # labels = labels.to_sparse()
        # labels = y
        #
        # inputs = {
        #     "input_ids": input_ids,
        #     "decoder_input_ids": decoder_input_ids
        #     # "labels": labels
        # }
        # the forward function automatically creates the correct decoder_input_ids

        student_distribution = self.calculate_student_model_distribution(x)
        loss = compute_loss(student_distribution, y, T, alpha)

        # loss = self.model(**inputs).compute_loss()
        # outputs = self.model(**inputs)
        # logits = outputs.logits
        # outputs = self.model(x)
        # loss.item()
        # logits = student_distribution

        # outputs = model(**encoded_input)
        # outputs = self.model(x)
        # logits = F.log_softmax(outputs, dim=1)
        # softmax_logits = F.softmax(logits, dim=1)
        # labels_expanded = labels[:, :, None]

        # output_teacher_batch = y
        # loss = torch.nn.KLDivLoss()(F.softmax(logits / T, dim=1),
        #                             F.softmax(labels_expanded / T, dim=1)) * (alpha * T * T) + \
        #        F.nll_loss(logits, labels) * (1. - alpha)

        # loss = torch.nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
        #                             F.softmax(output_teacher_batch / T, dim=1)) * (alpha * T * T) + \
        #        F.nll_loss(logits, y) * (1. - alpha)
        return loss, student_distribution


    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, logits = self._step(batch)
        # logits = self.calculate_student_model_distribution(x)
        acc, predicted_label = compute_accuracy(logits, y)
        self.log('train_loss_inbal', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        self.loss_values_list += [loss, acc, self.trainer.current_epoch, self.trainer.global_step]
        return {"loss": loss, "acc": acc}

    # def on_train_epoch_end(self, outputs):
    #     avg_loss = torch.cat([x['loss'].view(-1) for x in outputs]).mean()
    #     avg_acc = torch.cat([x['acc'].view(-1) for x in outputs]).mean()
    #
    #     print("--------------------")
    #     print("Train avg_loss: ", avg_loss)
    #     print("Train avg_acc: ", avg_acc)
    #     print("--------------------")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, logits = self._step(batch)
        logits = logits.squeeze(1)
        acc, predicted_label = compute_accuracy(logits, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return {
            "val_loss": loss,
            "val_acc": acc,
            "softmax_logits": logits.tolist(),
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

    # def prepare_data(self):
        # download
        # MNIST(self.other_arguments.data_dir, train=True, download=True)
        # MNIST(self.other_arguments.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # if stage == "fit" or stage is None:
        #     mnist_full = MNIST(self.other_arguments.data_dir, train=True, transform=self.transform)
        #     number_of_train_samples = 55000
        #
        #     if(self.other_arguments.max_train_samples != -1):
        #         number_of_train_samples = min(self.other_arguments.max_train_samples, 55000)
        #     self.mnist_train = torch.utils.data.Subset(mnist_full, [i for i in range(number_of_train_samples)])
        #     self.mnist_val = torch.utils.data.Subset(mnist_full, [i for i in range(55000, 60000)])
        #
        # if stage == "test" or stage is None:
        #     self.mnist_test = MNIST(self.other_arguments.data_dir, train=False, transform=self.transform)

        # mnist_full = MNIST(self.other_arguments.data_dir, train=True, transform=self.transform)
        # dataset = load_dataset("csv", data_files="final_dataset.csv")

        dataset = load_dataset("csv", data_files="final_dataset.csv")

        # dataset = dataset.remove_columns("Unnamed: 0")

        number_of_train_samples = len(dataset)
        if (self.other_arguments.max_train_samples != -1):
            number_of_train_samples = min(self.other_arguments.max_train_samples, number_of_train_samples)
        # self.mnist_train = torch.utils.data.Subset(dataset, [i for i in range(number_of_train_samples)])
        # self.mnist_val = MNIST(self.other_arguments.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        # dataloader = DataLoader(
        #     self.mnist_train,
        #     self.other_arguments.train_batch_size,
        #     drop_last=False, shuffle=True,
        #     num_workers=self.training_arguments.num_workers)
        # dataset = load_dataset("csv", data_files="final_dataset.csv")

        dataset = load_dataset("csv", data_files="final_dataset.csv")

        dataset = dataset["train"]
        # dataset = dataset.remove_columns(["idx", "task"])
        dataloader = DataLoader(dataset, self.other_arguments.train_batch_size, drop_last=False,
                                    shuffle=True, num_workers=self.training_arguments.num_workers)

        return dataloader

    # def val_dataloader(self):

        # return DataLoader(self.mnist_val,
        #                   batch_size=self.other_arguments.eval_batch_size,
        #                   num_workers=self.training_arguments.num_workers)

    def calculate_student_model_distribution(self, input):
        vocabulary = self.tokenizer.get_vocab()
        labels = list(vocabulary.keys())
        class_ids = torch.LongTensor(self.tokenizer(labels, padding="longest").input_ids)


        # class_ids = nn.ZeroPad2d((0, 200 - class_ids.size()[1]))(class_ids)


        encoding = self.tokenizer(input, return_tensors="pt", return_length=True)

        # encoding_ids = nn.ZeroPad2d((0, 200 - encoding.input_ids.size()[1]))(encoding.input_ids)
        # generated_outputs = self.model.generate(encoding_ids, do_sample=False, output_scores=True,
        #                                    return_dict_in_generate=True)


        generated_outputs = self.model.generate(encoding.input_ids, do_sample=True,num_beams=6, output_scores=True,
                                           return_dict_in_generate=True)
        while len(generated_outputs.scores) <= 4:
            generated_outputs = self.model.generate(encoding.input_ids, do_sample=False, output_scores=True,
                                                    return_dict_in_generate=True)

        # Generate the logits for each token in the generated output sequence.
        # `scores` has size [batch, seq_length, vocab_size]
        scores = torch.stack(generated_outputs.scores, dim=1)

        # transpose and expand to match the dimensions
        score_of_labels = scores.gather(dim=2, index=class_ids.T.expand(1, -1, -1))
        # probabilities = score_of_labels.nanmean(dim=1).softmax(1)
        probabilities = score_of_labels.softmax(2)
        # max_probability_index = torch.argmax(probabilities, dim=1)[0]

        # entailment = labels[max_probability_index]
        # probability = probabilities[0, max_probability_index].item()
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
    other_arguments.add_argument("--num_train_epochs", default=4, type=int)
    other_arguments.add_argument("--gradient_accumulation_steps", default=1, type=int)
    other_arguments.add_argument("--seed", default=42, type=int)
    other_arguments.add_argument("--save_top_k", default=-1, type=int)
    other_arguments.add_argument("--save_last", default=False, action="store_true")
    other_arguments.add_argument("--write_dev_predictions", default=False, action="store_true")
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
        accumulate_grad_batches=other_arguments.gradient_accumulation_steps,
        # gpus=training_arguments.n_gpu,
        deterministic=True,
        max_epochs=other_arguments.num_train_epochs,
        precision=16 if training_arguments.fp_16 else 32,
        # amp_level=training_arguments.opt_level,
        gradient_clip_val=training_arguments.max_grad_norm,
        callbacks=[checkpoint_callback],
        fast_dev_run=other_arguments.do_fast_dev_run,
    )

    if (other_arguments.limit_train_batches != -1):
        train_params["limit_train_batches"] = other_arguments.limit_train_batches

    if (other_arguments.limit_val_batches != -1):
        train_params["limit_val_batches"] = other_arguments.limit_val_batches

    if (training_arguments.distributed_backend != None):
        train_params["distributed_backend"] = training_arguments.distributed_backend

    # dataLoader = model.train_dataloader()

    trainer = Distilation_Trainer.DistilationTrainer(**train_params)
    # trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    values_list = model.loss_values_list
    # open file
    with open('loss_values.txt', 'w+') as f:
        # write elements of list
        for items in values_list:
            for value in items:
                f.write('%s,' % value)
            f.write('\n')
        print("File written successfully")
    f.close()

