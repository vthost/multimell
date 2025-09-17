import os
from dotenv import load_dotenv
# Load the environment variables from the .env file
# esp. do that before very first huggingface import!
load_dotenv()
print("HF_HOME", os.getenv("HF_HOME"))

import json
import copy
import yaml
from lm_eval.__main__ import setup_parser, parse_eval_args, cli_evaluate
from lm_eval.tasks import TaskManager, get_task_dict


def load_lmeval_dataset(task_name, model_args=""):

    metadata = {}
    # (
    #     simple_parse_args_string(model_args) # TODO we probably don't need that?
    #     if isinstance(model_args, str)
    #     else model_args
    #     if isinstance(model_args, dict)
    #     else {}
    # ) | (metadata or {}))
    task_manager = TaskManager(metadata=metadata)

    task_dict = get_task_dict(
        task_name,
        task_manager,
    )

    task_config = task_dict[task_name].config  # may use this to translate only those parts needed for evaluation
    dataset_dict = task_dict[task_name].dataset

    # this seems to be column names for split currently selected. but we may want to translate all?
    # features = task_dict[task_name].features  # relevant columns?
    return dataset_dict, task_config


def translate_dataset(dataset_dict, task_config):
    # TODO implement translate & verify... you can use these fields etc.:
    split_names = list(dataset_dict.column_names.keys())

    print("columns to translate:")
    for k, v in dataset_dict.column_names.items():
        print(k, v)

    for split_name in split_names:
        print("split:", split_name)
        dataset_split = dataset_dict[split_name]
        dataset_dict[split_name].set_format(type="pandas")

        for i in range(len(dataset_split)):
            df_row = dataset_split[i]
            print(df_row)
            if i > 3: break

    # return a simple dict, we later assume we've done the pd dataset formatting above
    # : 5 for testing
    return {split_name: dataset_dict[split_name][:5] for split_name in split_names}


def save_dataset(dataset_dict, data_path):

    os.makedirs(f"{data_path}", exist_ok=True)
    # TODO add some readme with creation data / dataset.json ?

    for split_name, dataset in dataset_dict.items():

        dataset.to_json('tmp_output.json', orient='records', indent=4)
        with open("tmp_output.json", "r", encoding="utf-8") as infile:
            data = json.load(infile)

        with open(f"{data_path}/{split_name}.jsonl", "w", encoding="utf-8") as outfile:
            for item in data:
                json.dump(item, outfile)
                outfile.write("\n")

        os.remove('tmp_output.json')


# TODO need to translate doc_to_target and doc_to_text (and possibly others) in task config
def create_new_task_config(orig_task_config, data_path, file_path, new_task_postfix, split_names):
    task_config = copy.deepcopy(orig_task_config)
    del task_config.dataset_name  # otherwise datasets loader throws BuilderException instead of using 'default' builder

    task_config.dataset_path = f'{data_path}' # TODO need correct punctuation ?
    task_config.dataset_kwargs["data_files"] = {f'{name}': f'{name}.jsonl' for name in split_names}

    task_config.task = f"{task_config.task}{new_task_postfix}"
    task_config.task_alias = f"{task_config.task_alias}{new_task_postfix}"
    task_config_dict = task_config.to_dict()  # TODO check if all was kept

    with open(file_path, 'w') as file:
        yaml.dump(task_config_dict, file, default_flow_style=False)


task_name = "mmlu_college_mathematics"
target_lang_id = "ger"

new_task_postfix = f"_{target_lang_id}"
new_task_name = f"{task_name}{new_task_postfix}"
new_task_data_path = f"./{new_task_name}"
new_task_config_path = f"./tasks/{new_task_name}/{new_task_name}.yaml"

os.makedirs("./tasks", exist_ok=True)
os.makedirs(f"./tasks/{new_task_name}", exist_ok=True)

dataset_dict, task_config = load_lmeval_dataset(task_name)

new_dataset_dict = translate_dataset(dataset_dict, task_config)

save_dataset(new_dataset_dict, new_task_data_path)

create_new_task_config(task_config, new_task_data_path, new_task_config_path, new_task_postfix, list(new_dataset_dict.keys()))

# Create an empty Namespace object
parser = setup_parser()
args = parse_eval_args(parser)
args.device = 'mps'
args.tasks = f"./tasks/{new_task_name}"
# args.limit = 2
args.output_path = "./lmeval_out/"  # save i/o
args.batch_size = 1
# MODEL_NAME = "google/flan-t5-small"
MODEL_NAME = "EleutherAI/pythia-14m,dtype=float32"
# MODEL_NAME = "google/gemma-2-2b-it"
# MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"  #"meta-llama/Llama-2-7b-hf"
args.model_args = f"pretrained={MODEL_NAME}"
args.model = 'hf'
args.gen_kwargs = ('return_dict_in_generate=True,output_hidden_states=True,output_logits=True,'
                   f'do_sample=False,temperature=0.0,min_new_tokens=2')
args.seed = [0, 1234, 1234, 1234]
args.log_samples = True
args.write_out = True  # log some dataset samples to stdout to double check
args.show_config = False  # otherwise would print out all hidden...
args.include_path = "./tasks/"  # include our task specifications

cli_evaluate(args)


# import mellea
# from mellea import start_session
# from mellea.backends.types import ModelOption
#
#
# base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-20b/v1"
# m = start_session(
#     backend_name="openai",
#     default_headers={"RITS_API_KEY": os.getenv("RITS_API_KEY")},
#     model_id="openai/gpt-oss-20b",
#     base_url=base_url,
#     model_options={ModelOption.SEED: 48},
# )
# print(m.chat("What is the etymology of mellea?").content)
