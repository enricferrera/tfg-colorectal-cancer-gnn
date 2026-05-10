import numpy as np
import torch

def calculate_class_weights(patient_dict):
    bt_dict = patient_dict
    patient_list = list(bt_dict.keys())
    label_list = []
    for pat in bt_dict:
        label_list.append(bt_dict[pat]['label'])

    unique_classes, class_counts = np.unique(label_list, return_counts=True)
    print("unique classes: ", unique_classes)
    total_samples = len(label_list)
    class_weights = []

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights.append(class_weight)

    f_weights = []
    tot = np.sum(class_weights)
    for weight in class_weights:
        weight = weight / tot
        f_weights.append(weight)

    weights = torch.tensor(f_weights, dtype=torch.float32)
    print("weights: ", weights)

    return weights