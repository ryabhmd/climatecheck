import os
import re
import json
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import krippendorff
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr, kendalltau


model_names = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "berkeley-nest/Starling-LM-7B-alpha",
    # "CohereForAI/c4ai-command-r-v01",
    # "CohereForAI/c4ai-command-r-plus",
    "allenai/OLMo-7B-0724-Instruct-hf",
    # "gemini-1.5-flash-latest",
    "Qwen2.5-VL-72B-Instruct",
    "zai-org/GLM-4.6"
]

veracity = {
    "supports": 0,
    "refutes": 1,
    "not_enough_info": 3,
    "not enough information": 3
}

def evaluate(set_h, set_m):
    # 1. correlation between human and model responses
    valid_set_h, valid_set_m = [], []
    for idx in range(len(set_h)):
        if set_m[idx] is not None:
            valid_set_h.append(set_h[idx])
            valid_set_m.append(set_m[idx])
    # try:
    correlation_p = pearsonr(valid_set_h, valid_set_m)
    correlation_s = spearmanr(valid_set_h, valid_set_m)
    correlation_k = kendalltau(valid_set_h, valid_set_m)
    # except ValueError:  # no valid responses from the model
    #     correlation_p, correlation_s, correlation_k = [[float("nan")] * 2] * 3
        
    try:
        kappa_score = cohen_kappa_score(valid_set_h, valid_set_m)
        if valid_set_h == valid_set_m:
            kappa_score = 1
    except ValueError:  # no kappa for numerical data
        kappa_score = float("nan")

    # 2. agreement between responses of humans
    # all_equal = True
    # for i in range(1, len(set_all_h)):
    #     if set_all_h[i] != set_all_h[i - 1]:
    #         all_equal = False
    #         break
    # if len(set_all_h) < 2:  # no agreement if only one rating per instance
    #     agreement = np.nan
    # elif all_equal:
    #     agreement = 1
    # agreement = krippendorff.alpha(
    #     reliability_data=set_all_h, level_of_measurement="nominal"
    # )
    return {
        "corr_coeff": {
            "pearson": round(float(correlation_p[0]), 4),
            "spearman": round(float(correlation_s[0]), 4),
            "kendall": round(float(correlation_k[0]), 4),
        },
        "p_value": {
            "pearson": round(float(correlation_p[1]), 4),
            "spearman": round(float(correlation_s[1]), 4),
            "kendall": round(float(correlation_k[1]), 4),
        },
        "kappa_score": round(kappa_score, 4),
        "total_responses": len(set_h),
        # "krippendorff_alpha": agreement
    }


if __name__ == "__main__":

    model = model_names[0]
    model_name = model.split("/")[1]
    with open(f"outputs/{model_name}.json", 'r') as file:
        claims_annotated = json.load(file)

    set_h, set_m = [], []
    for row in claims_annotated:
        if row['annotation'] and row['annotation'] != "One Annotation":
            set_h.append(veracity[row['annotation'].lower()])
            set_m.append(veracity[row['predicted_label'].lower()])

    print("===== INFO ======")
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    print("Evaluating annotation")
    print("Model: ", model_name)

    results = [evaluate(set_h, set_m)]

    with open("outputs/annotation_evaluation.json", "r") as f:
        all_results = json.load(f)

    all_results[model_name] = results
    with open("outputs/annotation_evaluation.json", "w") as f:
        json.dump(all_results, f)

