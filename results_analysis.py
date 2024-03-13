import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_pandas_from_results(filepath):
    pandas_table = pd.read_csv(filepath, delimiter="\|\|", index_col=False)
    return pandas_table


def evaluate_model_parameter_results(model_parameters_1, model_parameters_2, experiment1_name, experiment2_name):
    # Group by 'epoch' and calculate the average of 'loss', 'accuracy', and the difference between 'student_probability_for_true_label' and 'true_probability_of_label'
    result_df_1 = model_parameters_1.groupby('epoch').agg({'loss': 'mean', 'accuracy': 'mean',
                                         'student_probability_for_true_label': lambda x: (
                                                 model_parameters_1['true_probability_of_label'] - x).mean()})

    # Rename the lambda function result column to something meaningful
    result_df_1.rename(columns={'student_probability_for_true_label': 'avg_difference_student_true_prob'}, inplace=True)

    result_df_2 = model_parameters_2.groupby('epoch').agg({'loss': 'mean', 'accuracy': 'mean',
                                         'student_probability_for_true_label': lambda x: (
                                                 model_parameters_2['true_probability_of_label'] - x).mean()})

    # Rename the lambda function result column to something meaningful
    result_df_2.rename(columns={'student_probability_for_true_label': 'avg_difference_student_true_prob'}, inplace=True)

    draw_all_graphs_for_data(result_df_1, "Accuracy per Epoch" + experiment1_name, "Loss per Epoch" + experiment1_name,
                             "Difference Between True and Student\nProbabilities per Epoch" + experiment1_name)

    draw_all_graphs_for_data(result_df_2, "Accuracy per Epoch" + experiment2_name, "Loss per Epoch" + experiment2_name,
                             "Difference Between True and Student\nProbabilities per Epoch" + experiment2_name)


def draw_all_graphs_for_data(data_table, accuracy_title, loss_title, difference_title):
    draw_per_epoch_graph(data_table, 'accuracy', accuracy_title, 'Epochs', 'Accuracy')
    draw_per_epoch_graph(data_table, 'loss', loss_title, 'Epochs', 'Loss')
    draw_per_epoch_graph(data_table, 'avg_difference_student_true_prob', difference_title, 'Epochs',
                         'Difference Between\nTrue and Student Probabilities')


def draw_per_epoch_graph(data_table, column_name, plot_name, x_name, y_name):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(data_table.index, data_table[column_name], marker='o', linestyle='-')
    plt.xlabel(x_name, fontsize=20)
    plt.ylabel(y_name, fontsize=20)
    # plt.title(plot_name)
    plt.grid(True)
    plt.xticks(data_table.index, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()


def fix_data_separator(*file_paths):
    for file_path in file_paths:
        # Read the data line by line, replace commas, and then write the modified lines to a new file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Process each line and replace the commas
        modified_lines = []
        for line in lines:
            # Split the line by ","
            parts = line.split(",")
            # Combine the parts of the third column back together
            third_column = ",".join(parts[2:-4])
            # Replace the first two and last four commas with "||"
            modified_line = parts[0] + "||" + parts[1] + "||" + third_column + "||" + "||".join(parts[-4:])
            modified_lines.append(modified_line)

        # Write the modified lines to a new file
        # Split the path based on '/'
        parts = file_path.rsplit('/', 1)
        # Join the parts except the last one
        new_file_path = parts[0] + '/modified_model_run_parameters.txt'
        # new_file_path = "results/10Epochs_2Batches/modified_model_run_parameters.txt"
        with open(new_file_path, 'w') as file:
            file.writelines(modified_lines)

    print("All data successfully modified and saved to 'modified_data.txt'")


def batch_size_experiment():
    model_parameters_batch_2 = create_pandas_from_results("results/10Epochs_2Batches/modified_model_run_parameters.txt")
    model_parameters_batch_5 = create_pandas_from_results("results/10Epochs_5Batches/modified_model_run_parameters.txt")

    evaluate_model_parameter_results(model_parameters_batch_2, model_parameters_batch_5,
                                     " with Batch Size 2", " with Batch Size 5")


def epoch_number_experiment():
    model_parameters_epoch_5 = create_pandas_from_results("results/5Epochs_2Batches/modified_model_run_parameters.txt")
    model_parameters_epoch_10 = create_pandas_from_results("results/10Epochs_2Batches/modified_model_run_parameters.txt")

    evaluate_model_parameter_results(model_parameters_epoch_5, model_parameters_epoch_10, "", " ")


def compare_tasks_experiment():
    model_parameters_epoch_10 = create_pandas_from_results("results/10Epochs_2Batches/modified_model_run_parameters.txt")
    df_sentiment = model_parameters_epoch_10[model_parameters_epoch_10['input'].str.startswith("what sentiment the following sentence has:")]
    df_yes_no = model_parameters_epoch_10[model_parameters_epoch_10['input'].str.startswith("Answer yes or no:")]

    evaluate_model_parameter_results(df_sentiment, df_yes_no, " in Sentiment Analysis", " in Question Answering")


if __name__ == "__main__":
    # fix_data_separator("results/5Epochs_2Batches/model_run_parameters.txt",
    #                    "results/5Epochs_5Batches/model_run_parameters.txt",
    #                    "results/10Epochs_2Batches/model_run_parameters.txt",
    #                    "results/10Epochs_5Batches/model_run_parameters.txt")
    # epoch_number_experiment()
    batch_size_experiment()
    compare_tasks_experiment()
