import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

MAX_HITS = 10


@dataclass
class HitsMetric:
    total: int = 0
    hit1: int = 0
    hit3: int = 0
    hit10: int = 0
    mrr_sum: float = 0.0  # 添加MRR的累计和

    def update(self, rank):
        if rank <= 1:
            self.hit1 += 1
        if rank <= 3:
            self.hit3 += 1
        if rank <= 10:
            self.hit10 += 1
        self.mrr_sum += 1.0 / rank  # 添加倒数排名的值

    def dump(self):
        return {
            "total": self.total,
            "hit1": self.hit1 / self.total,
            "hit3": self.hit3 / self.total,
            "hit10": self.hit10 / self.total,
            "mrr": self.mrr_sum / self.total,  # 添加MRR指标（平均倒数排名）
        }
    
    def print_metric(self):
        """打印评估指标"""
        if self.total == 0:
            print("没有可用的评估样本！")
            return
        
        metrics = self.dump()
        print(f"总样本数: {self.total}")
        print(f"Hits@1: {metrics['hit1']:.4f}")
        print(f"Hits@3: {metrics['hit3']:.4f}")
        print(f"Hits@10: {metrics['hit10']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")
    
    def get_metric(self):
        """获取评估指标"""
        return self.dump()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen1.5b", type=str)
    parser.add_argument(
        "--dataset",
        choices=["ICEWS14", "ICEWS18", "WIKI", "YAGO"],
        default="ICEWS18",
        type=str,
    )
    parser.add_argument(
        "--multi_step", default=False, action="store_true"
    )  # inference in multi_step
    # History Modeling
    parser.add_argument(
        "--history_type",
        choices=["entity", "pair"],
        default="entity",
        type=str,
    )  # entity or pair
    parser.add_argument(
        "--history_direction",
        choices=["uni", "bi"],
        default="uni",
        type=str,
    )  # uni or bidirectional
    parser.add_argument("--history_len", default=0, type=int)  # length of history
    parser.add_argument(
        "--history_top_k", default=1, type=int
    )  # number of target to choose
    # Prompting
    parser.add_argument(
        "--label", 
        default=False, 
        action="store_true",
        help="是否在提示中使用标签形式。当为True时，提示格式为'[entity,relation,0.target]'；"
             "同时会影响candidates的格式为{0: 'target', 1: 'target2'}"
    )  
    parser.add_argument(
        "--text_style", 
        default=False, 
        action="store_true",
        help="是否使用文本形式。当为True时，会加载entity2id.txt和relation2id.txt将ID转换为文本"
    )  
    parser.add_argument(
        "--no_entity", 
        default=False, 
        action="store_true",
        help="是否在提示中省略实体名称。当为True时，提示格式为'[entity,relation,0]'；"
             "同时会影响candidates的格式为{0: 'target', 1: 'target2'}"
    )  
    parser.add_argument(
        "--sys_instruction", default="", type=str
    )  # system instruction for chatgpt
    parser.add_argument(
        "--no_time", default=False, action="store_true"
    )  # whether to not include time
    parser.add_argument(
        "--shuffle_history", default=False, action="store_true"
    )  # whether to include history
    # Active Learning
    parser.add_argument(
        "--active_learning", default=False, action="store_true"
    )  # whether to use active learning
    parser.add_argument(
        "--active_strategy",
        choices=["random", "max_entropy", "best_of_k", "random_balanced", "rl"],
        default="random",
        type=str,
    )  # active learning strategy
    parser.add_argument(
        "--active_samples", default=5, type=int
    )  # number of samples to select
    parser.add_argument(
        "--active_integration",
        choices=["direct", "labeled"],
        default="direct",
        type=str,
    )  # integration method for active samples
    
    # 强化学习相关参数
    parser.add_argument(
        "--rl_train", default=False, action="store_true"
    )  # 是否训练RL策略
    parser.add_argument(
        "--rl_train_size", default=500, type=int
    )  # 训练RL策略使用的样本数量
    parser.add_argument(
        "--rl_config", default="rl_configs/tkg-agent.yaml", type=str
    )  # RL配置文件路径
    parser.add_argument(
        "--rl_model_path", default=None, type=str
    )  # 预训练RL模型路径
    parser.add_argument(
        "--bert_model", default="/data/shangyuan/models/bert-base-chinese", type=str
    )  # BERT模型路径，用于强化学习中的文本编码
    
    # 离线RL训练相关参数
    parser.add_argument(
        "--offline_rl", default=False, action="store_true"
    )  # 是否使用离线RL训练模式
    parser.add_argument(
        "--offline_data_path", default=None, type=str
    )  # 离线数据文件路径
    parser.add_argument(
        "--offline_train_steps", default=10000, type=int
    )  # 离线训练的总优化步数
    parser.add_argument(
        "--cql_weight", default=0.0, type=float
    )  # CQL损失权重，设为0禁用CQL
    # 添加离线模型检查点路径参数
    parser.add_argument(
        "--offline_model_checkpoint_path", default=None, type=str
    )  # 离线训练好的模型检查点路径，用于阶段三（测试/策略应用）
    
    # 添加output_dir参数
    parser.add_argument(
        "--output_dir", default="./outputs", type=str
    )  # 输出目录路径
    
    # Hyperparams
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--dec_cand", default=5, type=int)  # num of candidate to decode
    parser.add_argument("--max_length", default=1, type=int)  # max length to decode
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--tokenizer_revision", default="main", type=str)
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--gpu", default=0, type=int)
    # Evaluation
    parser.add_argument(
        "--eval_filter",
        choices=["none", "static", "time-aware"],
        default="none",
        type=str,
    )

    args = parser.parse_args()
    assert args.label or not args.no_entity

    return args


# Read entity2id, relation2id
def load_dictionary(in_path: str, file_name: str) -> Dict[int, str]:
    _dict = {}
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split("\t")
            node = line_split[0]
            index = int(line_split[1])

            _dict[index] = node
    return _dict


# Read train, valid data to construct search space
def load_quadruples(
    search_dictionary: Dict[Any, Dict[Any, Dict[Any, List[Any]]]],
    in_path: str,
    file_name: str,
    entity_dictionary: Optional[Dict[int, str]] = None,
    relation_dictionary: Optional[Dict[int, str]] = None,
    query: str = "head",
):
    discard_line, total_line = 0, 0
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            total_line += 1
            line_split = line.split()
            if entity_dictionary and relation_dictionary:
                if (
                    int(line_split[0]) not in entity_dictionary
                    or int(line_split[2]) not in entity_dictionary
                    or int(line_split[1]) not in relation_dictionary
                ):
                    print(line)
                    discard_line += 1
                    continue
                head = entity_dictionary[int(line_split[0])]
                tail = entity_dictionary[int(line_split[2])]
                rel = relation_dictionary[int(line_split[1])]
            else:
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])

            time = int(line_split[3])

            if query == "head":
                if head not in search_dictionary:
                    search_dictionary[head] = {}
                if time not in search_dictionary[head]:
                    search_dictionary[head][time] = {}
                if rel not in search_dictionary[head][time]:
                    search_dictionary[head][time][rel] = []
                search_dictionary[head][time][rel].append(tail)
            elif query == "tail":
                if tail not in search_dictionary:
                    search_dictionary[tail] = {}
                if time not in search_dictionary[tail]:
                    search_dictionary[tail][time] = {}
                if rel not in search_dictionary[tail][time]:
                    search_dictionary[tail][time][rel] = []
                search_dictionary[tail][time][rel].append(head)

    print(f"# line discarded due to index issue: {discard_line} / {total_line}")


# Read test data to inferencee
def load_quadruples_for_test(
    in_path: str,
    file_name: str,
    entity_dictionary: Optional[Dict[int, str]] = None,
    relation_dictionary: Optional[Dict[int, str]] = None,
) -> List[List[Any]]:
    test_instances = []
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split()
            if entity_dictionary and relation_dictionary:
                if (
                    int(line_split[0]) not in entity_dictionary
                    or int(line_split[2]) not in entity_dictionary
                    or int(line_split[1]) not in relation_dictionary
                ):
                    print(line)
                    continue
                head = entity_dictionary[int(line_split[0])]
                tail = entity_dictionary[int(line_split[2])]
                rel = relation_dictionary[int(line_split[1])]
            else:
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
            time = int(line_split[3])
            test_instances.append((head, rel, tail, time))
    return test_instances


def format_data(data):
    tail_prediction, head_prediction = {}, {}
    for head, rel, tail, time in data:
        tail_key = (head, rel, time)
        if tail_key not in tail_prediction:
            tail_prediction[tail_key] = []
        tail_prediction[tail_key].append(tail)

        head_key = (tail, rel, time)
        if head_key not in head_prediction:
            head_prediction[head_key] = []
        head_prediction[head_key].append(head)

    formatted_data = list(
        sorted(
            [([k[0], k[1], list(set(v)), k[2]], "tail") for k, v in tail_prediction.items()]
            + [([k[0], k[1], list(set(v)), k[2]], "head") for k, v in head_prediction.items()],
            key=lambda x: x[0][3],
        )
    )
    return formatted_data


def load_data(args: argparse.Namespace, mode="test", global_entity_ids=None, global_relation_ids=None):
    """
    加载训练数据和搜索空间
    
    Args:
        args: 参数命名空间
        mode: 数据模式，"train"表示训练集，"valid"表示验证集，"test"表示测试集，默认为"test"
        global_entity_ids: 可选，全局实体ID映射，用于确保ID一致性
        global_relation_ids: 可选，全局关系ID映射，用于确保ID一致性
    
    Returns:
        formatted_data: 格式化后的数据集，格式为 [([实体, 关系, [目标列表], 时间戳], "head"或"tail")]
        head_search_space: 头实体搜索空间
        tail_search_space: 尾实体搜索空间
    """
    entity_dictionary, relation_dictionary = None, None
    # 1. 加载实体和关系映射(如果使用文本形式)
    if args.text_style:
        entity_dictionary = load_dictionary("/data/shangyuan/tkg_data/data", os.path.join(args.dataset, "entity2id.txt"))
        relation_dictionary = load_dictionary("/data/shangyuan/tkg_data/data", os.path.join(args.dataset, "relation2id.txt"))

    # 2. 构建搜索空间
    head_search_space = {} # 用于存储以头实体为索引的关系
    load_quadruples(
        head_search_space,
        "/data/shangyuan/tkg_data/data",
        os.path.join(args.dataset, "train.txt"),
        entity_dictionary,
        relation_dictionary,
        query="head",
    )
    load_quadruples(
        head_search_space,
        "/data/shangyuan/tkg_data/data",
        os.path.join(args.dataset, "valid.txt"),
        entity_dictionary,
        relation_dictionary,
        query="head",
    )

    tail_search_space = {} # 用于存储以尾实体为索引的关系
    load_quadruples(
        tail_search_space,
        "/data/shangyuan/tkg_data/data",
        os.path.join(args.dataset, "train.txt"),
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )
    load_quadruples(
        tail_search_space,
        "/data/shangyuan/tkg_data/data",
        os.path.join(args.dataset, "valid.txt"),
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )

    if args.history_direction == "bi":
        head_search_space.update(tail_search_space)
        tail_search_space = head_search_space

    # 3. 根据模式加载相应的数据集
    if mode == "test":
        # 加载测试数据
        data = load_quadruples_for_test(
            "/data/shangyuan/tkg_data/data",
            os.path.join(args.dataset, "test.txt"),
            entity_dictionary,
            relation_dictionary,
        )
    elif mode == "valid":
        # 加载验证数据
        data = load_quadruples_for_test(
            "/data/shangyuan/tkg_data/data",
            os.path.join(args.dataset, "valid.txt"),
            entity_dictionary,
            relation_dictionary,
        )
    elif mode == "train":
        # 加载训练数据
        data = load_quadruples_for_test(
            "/data/shangyuan/tkg_data/data",
            os.path.join(args.dataset, "train.txt"),
            entity_dictionary,
            relation_dictionary,
        )
    else:
        raise ValueError(f"不支持的模式: {mode}，必须是'train'、'valid'或'test'")

    # 4. 格式化数据
    formatted_data = format_data(data)

    return formatted_data, head_search_space, tail_search_space


def adjust_top_k(test_data, args):
    max_targets_len = max([len(x[0][2]) for x in test_data])
    args.top_k = max(args.top_k, MAX_HITS, max_targets_len + MAX_HITS)
    if args.verbose:
        print(f"max targets len: {max_targets_len}")
        print(f"adjusted top k: {args.top_k}")


def get_filename(args: argparse.Namespace, is_eval: bool = False):
    model_name = args.model.split("/")[-1]
    filename_args = "_".join(
        [
            model_name,
            args.dataset,
            f"multi_step_{args.multi_step}",
            f"history_len_{args.history_len}",
            f"history_type_{args.history_type}",
            f"history_direction_{args.history_direction}",
            f"no_time_{args.no_time}",
            f"shuffle_history_{args.shuffle_history}",
            f"label_{args.label}",
            f"text_style_{args.text_style}",
            f"no_entity_{args.no_entity}",
            f'world_size_{"*" if is_eval else args.world_size}',
            f'rank_{"*" if is_eval else args.rank}',
        ]
    )
    filename = f"outputs/{filename_args}.jsonl"
    print(f"output file: {filename}")
    return filename


def construct_history_by_search(
    search_space: Dict[str, Any], entity: str, relation: str, history_type: str
):
    if entity not in search_space:
        return {}

    search_graph = {entity: {}}

    if history_type == "entity":
        search_graph[entity] = search_space[entity]
    elif history_type == "pair":
        search_graph[entity] = {
            k: {relation: v[relation]} for k, v in search_space[entity].items() if relation in v
        }

    return search_graph


def format_history(
    history_graph: Dict[str, Any],
    history_len: int,
    question: List[str],
    args: argparse.Namespace,
    return_prompt: bool = True,
) -> Tuple[Union[List[tuple], List[List[Any]]], dict, list]:
    """格式化历史数据。

    如果 return_prompt=True, 返回: 结构化历史数据列表, 候选实体映射, 原始历史样本列表
    如果 return_prompt=False, 返回: 历史四元组列表, 候选实体映射, 原始历史样本列表
    
    候选实体映射 (candidates_dict) 根据 args.label 和 args.no_entity 确定格式，
    最终会被转换为索引形式 {0: entity1, 1: entity2, ...}。
    """
    quadruples = []
    selected_history_samples = []
    
    query_time = question[0]
    query_entity = question[1] if len(question) > 1 else None
    query_relation = question[2] if len(question) > 2 else None
    query_direction = question[-1] if len(question) > 3 and isinstance(question[-1], str) and question[-1] in ["head", "tail"] else None
    
    for entity in history_graph:
        for time in history_graph[entity]:
            if time >= query_time: continue
            for relation in history_graph[entity][time]:
                for target in history_graph[entity][time][relation]:
                    # 存储原始元组信息
                    sample_tuple = (entity, relation, [target], time) 
                    quadruples.append(sample_tuple)

    quadruples.sort(key=lambda x: x[3])
    selected_quadruples = quadruples[-history_len:]

    if args.shuffle_history:
        random.shuffle(selected_quadruples)

    # --- 修改开始 ---
    # 初始化返回值
    structured_history_data = [] # 用于 return_prompt=True
    history_list_for_output = [] # 用于 return_prompt=False
    candidates_dict = {}
    candidate_id = 0

    for h, r, t_list, ts in selected_quadruples:
        target = t_list[0] 
        
        # 填充两个列表
        structured_history_data.append((ts, h, r, target)) # 存储结构化数据
        history_list_for_output.append([h, r, target, ts]) # 存储旧格式列表

        # 填充 selected_history_samples (逻辑不变)
        hist_direction = "head" if query_direction == "tail" else ("tail" if query_direction == "head" else None)
        if hist_direction:
             selected_history_samples.append(((h, r, t_list, ts), hist_direction))

        # 构建 candidates_dict (逻辑基本不变, 基于历史目标)
        if args.label or args.no_entity: # 需要索引的情况
            if target not in candidates_dict.values():
                candidates_dict[candidate_id] = target
                candidate_id += 1
        else: # 不需要索引，键值都是实体
             if target not in candidates_dict:
                 candidates_dict[target] = target

    # --- 修改：确保返回的 candidates_dict 总是索引形式 ---
    final_indexed_candidates = {}
    if args.label or args.no_entity:
         final_indexed_candidates = candidates_dict # 已经是 {idx: entity}
    else:
         # 转换 {entity: entity} 为 {idx: entity}
         idx = 0
         # 保留某种固定顺序（例如，按实体名称排序）以确保一致性
         sorted_entities = sorted(list(candidates_dict.keys())) 
         for entity in sorted_entities:
              final_indexed_candidates[idx] = entity
              idx += 1
              
    # 根据 return_prompt 决定第一个返回值
    history_output = structured_history_data if return_prompt else history_list_for_output

    return history_output, final_indexed_candidates, selected_history_samples
    # --- 修改结束 ---


def prepare_input(x, entity_search_space, args, return_prompt: bool = True, active_samples=None):
    """准备模型的输入，并将主动采样样本的目标实体整合到候选列表中，使用简化提示格式"""
    entity, relation, targets, time = x
    
    history_graph = construct_history_by_search(
        entity_search_space, entity, relation, args.history_type
    )

    # --- 修改开始 ---
    # 1. 调用修改后的 format_history
    # history_data 是 structured_history_data 或 history_list_for_output
    # initial_candidates_dict 是基于历史的 {idx: entity} 映射
    history_data, initial_candidates_dict, historical_samples = format_history(
        history_graph, args.history_len, [time, entity, relation, targets], args, return_prompt=return_prompt
    )

    # 如果不需要返回提示 (例如，非 LLM 基线)
    if not return_prompt:
        # history_data 此时是 history_list_for_output (List[List[Any]])
        # initial_candidates_dict 仍然需要转换成值列表
        return history_data, list(initial_candidates_dict.values()), historical_samples

    # --- 到这里，return_prompt 必然为 True ---
    # history_data 此时是 structured_history_data (List[tuple])

    # 2. 整合主动采样样本到候选者 (逻辑不变，但基于 initial_candidates_dict)
    final_candidates_dict = initial_candidates_dict.copy()
    entity_to_idx_map = {entity: idx for idx, entity in final_candidates_dict.items()}
    final_candidate_entities = list(final_candidates_dict.values())
    
    active_sample_lines_to_add = [] 

    if active_samples and len(active_samples) > 0:
        for sample_tuple, _ in active_samples:
            s_entity, s_relation, s_targets, s_time = sample_tuple
            if not s_targets: continue
            s_target = s_targets[0] 

            # 检查并添加新实体到最终映射
            if s_target not in entity_to_idx_map:
                new_idx = len(final_candidates_dict) 
                final_candidates_dict[new_idx] = s_target 
                entity_to_idx_map[s_target] = new_idx 
                final_candidate_entities.append(s_target)
            
            # --- 格式化主动样本行 (使用最终映射) ---
            target_idx = entity_to_idx_map[s_target]
            
            if args.label:
                 formatted_target_str = f"{target_idx}. {s_target}"
            elif args.no_entity:
                 formatted_target_str = f"{target_idx}"
            else: 
                 formatted_target_str = s_target 

            line = ""
            if not args.no_time:
                line += f"{s_time}:"
            line += f"[{s_entity},{s_relation},{formatted_target_str}]"
            active_sample_lines_to_add.append(line)

    # 3. 格式化历史样本行 (使用最终映射)
    formatted_history_lines = []
    # history_data 现在是 structured_history_data: List[(ts, h, r, target)]
    for ts, h, r, target in history_data:
         # 从最终映射获取索引 (应该总能找到)
         target_idx = entity_to_idx_map.get(target, -1) 
         if target_idx != -1:
             if args.label:
                  formatted_target_str = f"{target_idx}. {target}"
             elif args.no_entity:
                  formatted_target_str = f"{target_idx}"
             else:
                  formatted_target_str = target
                  
             line = ""
             if not args.no_time: line += f"{ts}:"
             line += f"[{h},{r},{formatted_target_str}]"
             formatted_history_lines.append(line)
         else:
             # 这个情况理论上不应发生，因为 target 来自 initial_candidates_dict
             print(f"警告: 历史目标 '{target}' 未在最终映射中找到！")


    # 4. 组装最终 Prompt (逻辑不变，使用新生成的行列表)
    model_input_lines = []
    if active_sample_lines_to_add:
        model_input_lines.append("### Current background events:")
        model_input_lines.extend(active_sample_lines_to_add)
        model_input_lines.append("") # Add a blank line for separation
    if formatted_history_lines:
        model_input_lines.append("### Historical sequence events:")
        model_input_lines.extend(formatted_history_lines)

    current_query_line = ""
    if not args.no_time: current_query_line += f"{time}:"
    current_query_line += f"[{entity},{relation}," # 以逗号结尾
    model_input_lines.append(current_query_line)

    model_input = "\n".join(model_input_lines)
    
    # 返回最终结果
    return model_input, final_candidate_entities, historical_samples
    # --- 修改结束 ---


def update_history(x, entity_search_space, predictions, candidates, args):
    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    if args.verbose:
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )
    if args.multi_step:
        filtered_predictions = [candidates[x[0]] for x in predictions if x[0] in candidates]
        targets = filtered_predictions[: args.history_top_k]
    entity_search_space[entity][time][relation] += targets
    if args.verbose:
        print(f"history:\n{entity},{relation},{time} --> {targets}")
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )


def write_results(x, predictions, candidates, direction, writer, args):
    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    example = {
        "timestamp": time,
        "entity": entity,
        "relation": relation,
        "targets": targets,
        "direction": direction,
        "predictions": [candidates[x[0]] for x in predictions if x[0] in candidates],
    }
    writer.write(json.dumps(example) + "\n")

    if args.verbose:
        print(f"example:\n{json.dumps(example, indent=2)}")

    return example


def update_metric(example, metric, args):
    # +++ Add detailed logging +++
    print(f"--- DEBUG: update_metric --- ")
    print(f"DEBUG_METRIC_INPUT_PREDICTIONS (len={len(example['predictions'])}): {example['predictions'][:20]}...") # Log first 20 predictions
    print(f"DEBUG_METRIC_INPUT_TARGETS: {example['targets']}")
    print(f"DEBUG_METRIC_BEFORE: Total={metric.total}, Hit1={metric.hit1}, Hit3={metric.hit3}, Hit10={metric.hit10}, MRR_Sum={metric.mrr_sum:.4f}")
    # --- End Add --- 

    if args.verbose:
        print(f'predictions: {example["predictions"]}')
    for target in example["targets"]:
        # +++ Add detailed logging +++
        print(f"DEBUG_METRIC_TARGET: Checking target '{target}'")
        # --- End Add ---
        metric.total += 1
        try:
            index = example["predictions"].index(target) if target in example["predictions"] else -1
            # +++ Add detailed logging +++
            print(f"DEBUG_METRIC_INDEX: Found target '{target}' at index {index}")
            # --- End Add ---
        except ValueError:
             # Should not happen if `target in example["predictions"]` check is correct, but for safety
             index = -1
             print(f"DEBUG_METRIC_INDEX: Target '{target}' not found in predictions (exception). Index set to -1.")
             
        if index >= 0:
            # Calculate rank: number of predictions *before* the target that are *not* also targets.
            # This correctly handles cases where multiple targets are present in the prediction list.
            _predictions_before_target = example["predictions"][:index]
            _filtered_predictions_before = [
                p for p in _predictions_before_target if p not in example["targets"]
            ]
            rank = len(_filtered_predictions_before) + 1
            
            # +++ Add detailed logging +++
            print(f"DEBUG_METRIC_RANK: Calculated rank for '{target}' is {rank}")
            # --- End Add ---
            
            if args.verbose:
                print(f"target: {target} --> rank: {rank}")
            metric.update(rank)
            # +++ Add detailed logging +++
            print(f"DEBUG_METRIC_AFTER_UPDATE ({target}): Total={metric.total}, Hit1={metric.hit1}, Hit3={metric.hit3}, Hit10={metric.hit10}, MRR_Sum={metric.mrr_sum:.4f}")
            # --- End Add ---
        else:
             # +++ Add detailed logging +++
             print(f"DEBUG_METRIC_RANK: Target '{target}' not found in predictions (index={index}). Rank not calculated.")
             print(f"DEBUG_METRIC_AFTER_MISS ({target}): Total={metric.total}, Hit1={metric.hit1}, Hit3={metric.hit3}, Hit10={metric.hit10}, MRR_Sum={metric.mrr_sum:.4f}")
             # --- End Add ---


def load_all_entity_relation_mappings(args: argparse.Namespace):
    """
    加载所有数据集(训练、验证、测试)的实体和关系映射，用于确保ID一致性
    
    Args:
        args: 参数命名空间
        
    Returns:
        entity_ids: 全局实体ID映射
        relation_ids: 全局关系ID映射
    """
    entity_dictionary, relation_dictionary = None, None
    
    # 如果使用文本形式，直接加载映射
    if args.text_style:
        entity_dictionary = load_dictionary("/data/shangyuan/tkg_data/data", os.path.join(args.dataset, "entity2id.txt"))
        relation_dictionary = load_dictionary("/data/shangyuan/tkg_data/data", os.path.join(args.dataset, "relation2id.txt"))
        
        # 转换为ID到实体名称的映射
        entity_ids = {v: k for k, v in entity_dictionary.items()}
        relation_ids = {v: k for k, v in relation_dictionary.items()}
        return entity_ids, relation_ids
    
    # 否则，需要从数据文件中提取所有唯一的实体和关系
    entity_ids = {}
    relation_ids = {}
    
    # 加载训练、验证和测试数据的所有四元组
    for file_name in ["train.txt", "valid.txt", "test.txt"]:
        file_path = os.path.join("/data/shangyuan/tkg_data/data", args.dataset, file_name)
        
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，跳过")
            continue
            
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line_split = line.split()
                
                if len(line_split) < 4:
                    print(f"警告：行格式不正确: {line}")
                    continue
                
                head = int(line_split[0])
                relation = int(line_split[1])
                tail = int(line_split[2])
                
                if head not in entity_ids:
                    entity_ids[head] = head
                    
                if tail not in entity_ids:
                    entity_ids[tail] = tail
                    
                if relation not in relation_ids:
                    relation_ids[relation] = relation
    
    print(f"从所有数据集创建全局ID映射：{len(entity_ids)}个实体，{len(relation_ids)}个关系")
    return entity_ids, relation_ids


def get_entities(data):
    """从数据集中提取所有唯一的实体
    
    Args:
        data: 数据集，格式为 [([实体, 关系, [目标列表], 时间戳], 方向), ...]
        
    Returns:
        所有唯一实体的集合
    """
    entities = set()
    for (sample, _) in data:
        entity, _, targets, _ = sample
        entities.add(entity)
        for target in targets:
            entities.add(target)
    return entities

def get_relations(data):
    """从数据集中提取所有唯一的关系
    
    Args:
        data: 数据集，格式为 [([实体, 关系, [目标列表], 时间戳], 方向), ...]
        
    Returns:
        所有唯一关系的集合
    """
    relations = set()
    for (sample, _) in data:
        _, relation, _, _ = sample
        relations.add(relation)
    return relations

def get_entity_relation_mappings(data):
    """从数据集中提取实体和关系的映射
    
    Args:
        data: 数据集，格式为 [([实体, 关系, [目标列表], 时间戳], 方向), ...]
        
    Returns:
        (entity_ids, relation_ids): 实体ID和关系ID的映射字典
    """
    entity_ids = {}
    relation_ids = {}
    
    for (sample, _) in data:
        entity, relation, targets, _ = sample
        
        if entity not in entity_ids:
            entity_ids[entity] = len(entity_ids)
            
        if relation not in relation_ids:
            relation_ids[relation] = len(relation_ids)
            
        for target in targets:
            if target not in entity_ids:
                entity_ids[target] = len(entity_ids)
                
    return entity_ids, relation_ids

def save_global_id_mappings(entity_ids, relation_ids, output_dir="./outputs"):
    """保存全局ID映射到文件
    
    Args:
        entity_ids: 实体ID映射
        relation_ids: 关系ID映射
        output_dir: 输出目录
    """
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "entity_ids.json"), "w", encoding="utf-8") as f:
        json.dump(entity_ids, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(output_dir, "relation_ids.json"), "w", encoding="utf-8") as f:
        json.dump(relation_ids, f, ensure_ascii=False, indent=2)
