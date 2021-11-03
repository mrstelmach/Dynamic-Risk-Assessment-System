# -*- coding: utf-8 -*-

"""
Module for model deployment.
"""

import json
import os
from shutil import copyfile

with open('config.json','r') as f:
    config = json.load(f)
dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def store_model_into_pickle(model_path, score_path, recor_path, target_dir):
    """
    Function copies the latest pickle file, the latestscore.txt value
    and the ingestfiles.txt file into the target_dir directory.
    """
    for path in [model_path, score_path, recor_path]:
        base_name = os.path.basename(path)
        copyfile(path, os.path.join(target_dir, base_name))


if __name__ == "__main__":
    store_model_into_pickle(
        os.path.join(output_model_path, 'trainedmodel.pkl'),
        os.path.join(output_model_path, 'latestscore.txt'),
        os.path.join(dataset_csv_path, 'ingestedfiles.txt'),
        prod_deployment_path
    )
