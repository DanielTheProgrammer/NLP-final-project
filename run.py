import pytorch_lightning as pl
import argparse
from transformers import T5ForConditionalGeneration
import knowledge_distilation
import teacher_dataset


def run():
    pl.seed_everything(other_arguments.seed)
    dataloader = create_dataloader()
    student_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = knowledge_distilation.ClassificationModelKD(training_arguments=training_arguments,
                                                        model_arguments=model_arguments,
                                                        other_arguments=other_arguments,
                                                        student_model=student_model,
                                                        dataloader=dataloader)

    train_params = dict(
        max_epochs=other_arguments.num_train_epochs,
        precision=16 if training_arguments.fp_16 else 32,
    )

    if other_arguments.limit_train_batches != -1:
        train_params["limit_train_batches"] = other_arguments.limit_train_batches

    if other_arguments.limit_val_batches != -1:
        train_params["limit_val_batches"] = other_arguments.limit_val_batches

    if training_arguments.distributed_backend != None:
        train_params["distributed_backend"] = training_arguments.distributed_backend

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


def create_dataloader():
    dataloader = teacher_dataset.create_dataloader("final_dataset.csv", other_arguments.train_batch_size)
    return dataloader


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
    other_arguments.add_argument("--num_train_epochs", default=5, type=int)
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

    run()
