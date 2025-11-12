import os
import sys
import json
import copy
import time
import torch
import hashlib
import argparse
from tqdm import tqdm
from termcolor import colored

from src.data import (
    GSM8K, MMLU, 
    WikiMultihopQA, StrategyQA, ComplexWebQuestions, QuasarT,
    HotpotQA, FEVER
)
from src.models.agents import NlAgent, CipherAgent, SDEAgent
from src.root_path import ROOT_PATH, DATA_ROOT_PATH
from src.utils import load_model, load_only_generation_config
from src.tasks import method_to_task_cls, update_hidden_dim


def generate_args_hash(
    setting: dict, 
    ignore_keys=["config_file", "sample", "redo", "sde_layer_config", "output_dir"],
):
    setting = {k: v for k, v in setting.items() if k not in ignore_keys}
    setting = json.dumps(setting, sort_keys=True)
    hash_output = hashlib.md5(setting.encode()).hexdigest()
    return hash_output


def main(args):
    if args.dataset == "gsm8k":
        dataset = GSM8K(data_root_path=DATA_ROOT_PATH)
    elif args.dataset.startswith("mmlu_"):
        field_name = args.dataset.split("mmlu_")[1]
        dataset = MMLU(data_root_path=DATA_ROOT_PATH, field_name=field_name)
    elif args.dataset == "2wqa":
        dataset = WikiMultihopQA(data_root_path=DATA_ROOT_PATH, retrieval_topk=args.retrieval_topk)
    elif args.dataset == "strategyqa":
        dataset = StrategyQA(data_root_path=DATA_ROOT_PATH, retrieval_topk=args.retrieval_topk)
    elif args.dataset == "cwq":
        dataset = ComplexWebQuestions(data_root_path=DATA_ROOT_PATH, retrieval_topk=args.retrieval_topk)
    elif args.dataset == "quasart":
        dataset = QuasarT(data_root_path=DATA_ROOT_PATH, retrieval_topk=args.retrieval_topk)
    elif args.dataset == "hotpotqa":
        dataset = HotpotQA(data_root_path=DATA_ROOT_PATH)
    elif args.dataset == "fever":
        dataset = FEVER(data_root_path=DATA_ROOT_PATH)
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    if args.generation_setting == "greedy":
        generation_config = dict(
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        cipher_generation_config = [dict(temperature=0.0) for i in range(args.agent_cnt)]
    else:
        generation_config = dict()
        default_generation_config = load_only_generation_config(args.model_name_or_path)
        cipher_generation_config = []
        for i in range(args.agent_cnt):
            cipher_generation_config.append(dict(temperature=default_generation_config["temperature"] * (i + 1) / args.agent_cnt))
    
    # create output file
    setting_dir = generate_args_hash(vars(args), ignore_keys=["output_dir", "resume", "redo", "resume_from", "sample", "test_layer"])
    model_save_name = args.model_name_or_path.split("/")[-1]
    if len(model_save_name) == 0:
        model_save_name = args.model_name_or_path.split("/")[-2]
    output_dir = os.path.join(ROOT_PATH, args.output_dir, args.task_type, args.dataset, model_save_name, args.method, setting_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.redo:
        run_id = len(os.listdir(output_dir))
    else:
        run_id = max(0, len(os.listdir(output_dir)) - 1) 
    output_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as fout:
        json.dump(vars(args), fout, indent=4)
    detail_file = os.path.join(output_dir, "detail.json")
    result_file = os.path.join(output_dir, "result.json")
    print(colored(f"### Output dir: {output_dir} ###", "green"))

    if os.path.exists(detail_file):
        with open(detail_file, "r") as fin:
            details = json.load(fin)
    else:
        details = []
    start_from = len(details)
    if (args.sample != -1 and start_from >= args.sample) or \
       (args.sample == -1 and start_from == len(dataset.dataset)): # if enough, skip, generate result
        if args.sample != -1:
            details = details[:args.sample]
        print(colored(f"### Already finished {start_from} samples, skip ###", "green"))
        print(colored(f"### Total sampled {len(details)} ###", "green"))
        task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
        run_task = task_cls([], dataset, args)
    else:
        model, tokenizer, model_config = load_model(args.model_name_or_path, args.method)
        update_hidden_dim(model_config.hidden_size)
        need_agent_cnt = args.agent_cnt
        agents = []
        for idx in range(need_agent_cnt):
            if args.method == "nl": 
                agent_cls = NlAgent
            elif args.method == "cipher":
                agent_cls = CipherAgent
            elif args.method == "sde":
                agent_cls = SDEAgent
            else:
                raise ValueError(f"Invalid method {args.method}")
            agents.append(agent_cls(
                engine_model_name_or_path=args.model_name_or_path,
                engine_model=model,
                engine_tokenizer=tokenizer,
                generation_configs=cipher_generation_config[idx] if args.method == "cipher" else generation_config,
                max_new_tokens=args.max_new_tokens,
                role_prompt=None,
            ))

        dataset.sample(sample_cnt=args.sample)
        task_cls = method_to_task_cls(task_type=args.task_type, method=args.method)
        run_task = task_cls(agents, dataset, args)
        pbar = tqdm(total=len(dataset.dataset) - len(details))
        for data in dataset.dataset[start_from:]:
            detail = run_task.run(data)
            detail["test_id"] = data["test_id"]
            details.append(detail)
            pbar.update(1)
            with open(detail_file, "w") as fout:
                json.dump(details, fout, indent=4)
        pbar.close()
    
    res = vars(args)
    res.update(run_task.generate_result(details))
    with open(result_file, "w") as fout:
        json.dump(res, fout, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--method", type=str, choices=["nl", "cipher", "sde"], required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--redo", action="store_true") # if True, do a new run
    
    # args in config file
    parser.add_argument("--task_type", type=str, choices=["ia", "debate", "workflow"])
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--agent_cnt", type=int)
    parser.add_argument("--generation_setting", type=str, choices=["default", "greedy"])
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--prompt_file", type=str)
    parser.add_argument("--retrieval_topk", type=int) # for IA, retrieval passage number

    # args for sde
    parser.add_argument("--edit_layer_idx", type=str, default=None) # such as "1" or "1,2,3,4"
    parser.add_argument("--sde_layer_config", type=str, default="configs/sde_layer.json") # default setting 
    
    args = parser.parse_args()

    if args.config_file is not None:
        try:
            with open(args.config_file, "r") as fin:
                config = json.load(fin)
            if config["task_type"] == "workflow":
                config["max_new_tokens"] = config["max_new_tokens_ma"]  
                del config["max_new_tokens_single"]
                del config["max_new_tokens_ma"]
            for k, v in config.items():
                if k not in vars(args) or getattr(args, k) is None:
                    setattr(args, k, v)
        except:
            assert False, "Invalid config file"

    if args.method == "sde":
        if args.edit_layer_idx is None:
            with open(args.sde_layer_config, "r") as fin:
                cfg = json.load(fin)
                args.edit_layer_idx = cfg[args.model_name_or_path]
        edit_layer_idx = args.edit_layer_idx.split(",")
        edit_layer_idx = [int(x) for x in edit_layer_idx] 
        args.edit_layer_idx = sorted(edit_layer_idx)
    
    if args.task_type == "debate" and args.sample is None:
        if args.dataset == "gsm8k":
            args.sample = 300
        else:
            args.sample = -1
    
    assert args.sample is not None

    print(args)
    main(args)