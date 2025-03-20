import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List
from dataclasses import dataclass, field

@dataclass
class AggregatedTrainingResults:
    models_training_accuracies: Dict[str, List[float]] = field(default_factory=dict)
    models_validation_accuracies: Dict[str, List[float]] = field(default_factory=dict)
    models_training_losses: Dict[str, List[float]] = field(default_factory=dict)
    models_validation_losses: Dict[str, List[float]] = field(default_factory=dict)
    
# TODO: modify to get the per class accuracy
def aggregate_training_results(folds_histories):
    # NOTE: For each model we are going to put a plot, a box plot or something to display the variance of the validation accuracy between each fold
    models_training_accuracies = {}
    models_validation_accuracies = {}

    for fold_history in folds_histories:
        for extractor_name, details in fold_history.items():
            if models_validation_accuracies.get(extractor_name) is None:
                models_validation_accuracies[extractor_name] = [details["best_validation_accuracy"]] 
            else:
                models_validation_accuracies[extractor_name].append(details["best_validation_accuracy"])
                
            if models_training_accuracies.get(extractor_name) is None:
                models_training_accuracies[extractor_name] = [details["best_training_accuracy"]] 
            else:
                models_training_accuracies[extractor_name].append(details["best_training_accuracy"])
                
    models_training_losses = {}
    models_validation_losses = {}

    for fold_history in folds_histories:
        for extractor_name, details in fold_history.items():
            if models_validation_losses.get(extractor_name) is None:
                models_validation_losses[extractor_name] = [details["best_validation_loss"]] 
            else:
                models_validation_losses[extractor_name].append(details["best_validation_loss"])
                
            if models_training_losses.get(extractor_name) is None:
                models_training_losses[extractor_name] = [details["best_training_loss"]] 
            else:
                models_training_losses[extractor_name].append(details["best_training_loss"])
            
            
    return AggregatedTrainingResults(
        models_training_accuracies=models_training_accuracies,
        models_validation_accuracies=models_validation_accuracies,
        models_training_losses=models_training_losses,
        models_validation_losses=models_validation_losses,
    )    
    
from experiments.helpers.plots_formatting import ICML

from enum import IntFlag, auto

class Metric(IntFlag):
    LOSS = auto()
    ACCURACY = auto()
    
    ALL= ACCURACY | LOSS
    
def plot_models_performance(aggregated_training_results: AggregatedTrainingResults, metric: Metric):
    plt.figure(figsize=(10, 5))
    
    plt.figure(figsize=ICML.double_column_figure() if Metric.ALL in metric else ICML.single_column_figure())
        
    models_validation_accuracies = aggregated_training_results.models_validation_accuracies
    models_validation_losses = aggregated_training_results.models_validation_losses

    if Metric.ACCURACY in metric:
        plt.subplot(1, 2 if Metric.ALL in metric else 1, 1)
        plt.boxplot(models_validation_accuracies.values(), labels=models_validation_accuracies.keys())
        plt.ylabel("Accuracy")
        plt.xticks(rotation=90)
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% Accuracy')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(0, 1)

    if Metric.LOSS in metric:
        plt.subplot(1, 2 if Metric.ALL in metric else 1, 2 if Metric.ALL in metric else 1)
        plt.boxplot(models_validation_losses.values(), labels=models_validation_losses.keys())
        plt.ylabel("Loss")
        plt.xticks(rotation=90)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(0, 1.5)
        
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

class Metric(IntFlag):
    LOSS = auto()
    ACCURACY = auto()
    ALL = ACCURACY | LOSS

def plot_models_comparison(aggregated_training_results: AggregatedTrainingResults, metric: Metric):
    models_validation_accuracies = aggregated_training_results.models_validation_accuracies
    models_training_accuracies = aggregated_training_results.models_training_accuracies
    models_validation_losses = aggregated_training_results.models_validation_losses
    models_training_losses = aggregated_training_results.models_training_losses

    models_names = models_validation_accuracies.keys()

    markers = list(Line2D.markers.keys())[:len(models_names)]

    plt.figure(figsize=ICML.double_column_figure() if Metric.ALL in metric else ICML.single_column_figure())

    if Metric.ACCURACY in metric:
        plt.subplot(1, 2 if Metric.ALL in metric else 1, 1)
        for model_name, marker in zip(models_names, markers):
            model_average_validation_accuracy = np.mean(models_validation_accuracies[model_name])
            model_average_training_accuracy = np.mean(models_training_accuracies[model_name])

            plt.scatter(
                model_average_validation_accuracy,
                model_average_training_accuracy,
                label=model_name,
                marker=marker,
                alpha=0.75,
                s=100
            )

        plt.xlabel("Validation Accuracy")
        plt.ylabel("Training Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

    if Metric.LOSS in metric:
        plt.subplot(1, 2 if Metric.ALL in metric else 1, 2 if Metric.ALL in metric else 1)
        for model_name, marker in zip(models_names, markers):
            model_average_validation_loss = np.mean(models_validation_losses[model_name])
            model_average_training_loss = np.mean(models_training_losses[model_name])

            plt.scatter(
                model_average_validation_loss,
                model_average_training_loss,
                label=model_name,
                marker=marker,
                alpha=0.75,
                s=100
            )

        plt.xlabel("Validation Loss")
        plt.ylabel("Training Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
def print_models_performances(aggregated_training_results: AggregatedTrainingResults, metric: Metric):
    models_validation_accuracies = aggregated_training_results.models_validation_accuracies
    models_validation_losses = aggregated_training_results.models_validation_losses
    
    if Metric.ACCURACY in metric:
        for model_name, accuracies in models_validation_accuracies.items():
            average_accuracy = np.mean(accuracies)
            min_accuracy, max_accuracy = np.min(accuracies), np.max(accuracies)
            std = np.std(accuracies)
            print(f"[{model_name}-accuracy]: {average_accuracy:.2f} ± {std:.2f}; ({min_accuracy:.2f}, {max_accuracy:.2f})")

        print(f"--- --- --- ---")

    if Metric.LOSS in metric:
        for model_name, losses in models_validation_losses.items():
            average_loss = np.mean(losses)
            min_loss, max_loss = np.min(losses), np.max(losses)
            std = np.std(losses)
            print(f"[{model_name}-loss]: {average_loss:.2f} ± {std:.2f}; ({min_loss:.2f}, {max_loss:.2f})")
            
        
def get_best_performing_fold(aggregated_training_results: AggregatedTrainingResults, metric: Metric) -> int:
    if metric == Metric.ACCURACY:
        # Compute the average accuracy per fold
        fold_accuracies = [
            np.mean([values[i] for values in aggregated_training_results.models_validation_accuracies.values()])
            for i in range(len(next(iter(aggregated_training_results.models_validation_accuracies.values()))))
        ]
        return np.argmax(fold_accuracies)  # Best accuracy = highest value

    elif metric == Metric.LOSS:
        # Compute the average loss per fold
        fold_losses = [
            np.mean([values[i] for values in aggregated_training_results.models_validation_losses.values()])
            for i in range(len(next(iter(aggregated_training_results.models_validation_losses.values()))))
        ]
        return np.argmin(fold_losses)  # Best loss = lowest value

    else:
        raise ValueError("Invalid metric. Choose Metric.ACCURACY or Metric.LOSS.")
    
import pandas as pd

from bouldering_video_segmentation.extractors import FeatureExtractor
    
def create_performance_table(aggregated_training_results: AggregatedTrainingResults, extractors: list[FeatureExtractor], latex: bool=False) -> pd.DataFrame:
    models_validation_accuracies = aggregated_training_results.models_validation_accuracies
    
    data = []
    
    model_keys = list(models_validation_accuracies.keys())
    
    for i, model_name in enumerate(model_keys):
        accuracies = models_validation_accuracies[model_name]
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        # Format the accuracy with standard deviation
        if latex:
            accuracy_with_std = f"{(avg_accuracy * 100):.2f}\% ± {(std_accuracy * 100):.2f}"
        else:
            accuracy_with_std = f"{(avg_accuracy * 100):.2f}% ± {(std_accuracy * 100):.2f}"
        
        # Create a row for this model
        row = {
            "Backbone Name": extractors[i].get_name(),
            "Backbone Type": extractors[i].get_features_type() if not latex else extractors[i].get_features_type().replace("FRAME_BY_FRAME","By Frame").replace("TEMPORAL", "By Segment"),
            "Model": "MLP",
            "Accuracy": accuracy_with_std
        }
        
        data.append(row)
    
    dataframe = pd.DataFrame(data)
    
    return dataframe

SEPARATION_LINE = "<THIS IS A SEPARATION LINE>"

get_separation_line_string = lambda number_of_columns: " & ".join([SEPARATION_LINE] * number_of_columns) + " \\\\"

def write_latex_table(dataframe, path, caption, small:bool=True, exact_position:bool=False, columns_alignments:str|list[str]=None):
    if isinstance(columns_alignments, str):
        columns_alignments = [columns_alignments] * len(dataframe.columns)
        
    if len(columns_alignments) != len(dataframe.columns):
        raise ValueError("The number of columns alignments must match the number of columns in the dataframe.")
    
    latex_code = dataframe.to_latex(
        index=False,
        caption=caption,
        column_format="".join(columns_alignments)
    )
    
    number_of_columns = len(dataframe.columns)

    if small:
        latex_code = latex_code.replace("\\begin{table}", "\\begin{table}\n\\centering\n\\small")
    else:
        latex_code = latex_code.replace("\\begin{table}", "\\begin{table}\n\\center")
        
    if exact_position:
        latex_code = latex_code.replace("\\begin{table}", "\\begin{table}[!h]")
                                        
    latex_code = latex_code.replace("\\caption{%s}" % caption, "")
    latex_code = latex_code.replace("\\end{table}", "\\vspace{-2ex}\\caption{%s}\n\\end{table}" % caption)
    
    latex_code = latex_code.replace("\\begin{tabular}", "\\resizebox{1\linewidth}{!}{\n\\begin{tabular}")
    latex_code = latex_code.replace("\\end{tabular}", "\end{tabular}\n}")
    
    latex_code = latex_code.replace(get_separation_line_string(number_of_columns), "\\midrule")

    with open(path, "w") as file:
        file.write(latex_code)