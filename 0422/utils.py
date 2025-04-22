import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

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
) -> Tuple[str, dict, list]:
    """格式化历史数据为提示字符串、候选实体映射和原始历史样本列表

    根据args.label和args.no_entity的设置:
    1. 当args.label=True时:
       - 提示格式: "[entity,relation,0.target]"
       - candidates格式: {0: "target", 1: "target2"}
    2. 当args.no_entity=True时:
       - 提示格式: "[entity,relation,0]"
       - candidates格式: {0: "target", 1: "target2"}
    3. 当两者都为False时:
       - 提示格式: "[entity,relation,target]"
       - candidates格式: {"target": "target"}

    注意：只有当args.model不是"recency"或"frequency"时，上述格式才会生效
    
    Returns:
        history_prompt: 格式化后的历史提示字符串
        candidates: 候选实体映射
        selected_history_samples: 选择用于提示的原始历史样本列表 [(tuple, direction), ...]
    """
    quadruples = []
    selected_history_samples = []
    
    # 确定查询时间戳 - 修正为从question[0]获取时间戳
    query_time = question[0]
    query_entity = question[1] if len(question) > 1 else None
    query_relation = question[2] if len(question) > 2 else None
    # 处理query_direction，默认为None，可能在question[-1]是字符串的情况下是方向
    query_direction = question[-1] if len(question) > 3 and isinstance(question[-1], str) and question[-1] in ["head", "tail"] else None
    
    # 遍历历史图谱中的实体
    for entity in history_graph:
        # 遍历该实体相关的时间戳
        for time in history_graph[entity]:
            # 跳过时间戳大于等于查询时间戳的记录
            if time >= query_time:
                continue
            # 遍历该时间戳下的关系
            for relation in history_graph[entity][time]:
                # 遍历该关系下的目标实体列表
                for target in history_graph[entity][time][relation]:
                    # 临时处理：假设我们无法可靠确定历史样本方向，存储为 None
                    sample_tuple = (entity, relation, [target], time)
                    sample_direction = None

                    quadruples.append(sample_tuple)

    # 根据时间戳对历史事件排序
    quadruples.sort(key=lambda x: x[3])

    # 选择最近的 history_len 个历史事件
    selected_quadruples = quadruples[-history_len:]

    # 如果设置了随机打乱历史顺序
    if args.shuffle_history:
        random.shuffle(selected_quadruples)

    # 格式化选中的历史事件为字符串，并构建候选实体映射
    history_prompt = ""
    candidates = {}
    candidate_id = 0

    if return_prompt and args.model not in ["recency", "frequency"]:
        for h, r, t_list, ts in selected_quadruples:
            target = t_list[0]
            
            hist_direction = "head" if query_direction == "tail" else ("tail" if query_direction == "head" else None)
            if hist_direction:
                 selected_history_samples.append(((h, r, t_list, ts), hist_direction))
            else:
                 pass

            event_str = ""
            if not args.no_time:
                event_str += f"{ts}:"

            if args.label:
                if target not in candidates.values():
                    candidates[candidate_id] = target
                    event_str += f"[{h},{r},{candidate_id}. {target}]"
                    candidate_id += 1
                else:
                    existing_id = list(candidates.keys())[list(candidates.values()).index(target)]
                    event_str += f"[{h},{r},{existing_id}. {target}]"
            elif args.no_entity:
                if target not in candidates.values():
                    candidates[candidate_id] = target
                    event_str += f"[{h},{r},{candidate_id}]"
                    candidate_id += 1
                else:
                    existing_id = list(candidates.keys())[list(candidates.values()).index(target)]
                    event_str += f"[{h},{r},{existing_id}]"
            else:
                candidates[target] = target
                event_str += f"[{h},{r},{target}]"
            
            history_prompt += event_str + "\n"

    return history_prompt, candidates, selected_history_samples


def prepare_input(x, entity_search_space, args, return_prompt: bool = True):
    """准备模型的输入

    Args:
        x: 当前样本元组 (entity, relation, targets, timestamp)
        entity_search_space: 相关实体的搜索空间
        args: 参数命名空间
        return_prompt: 是否返回格式化后的提示字符串

    Returns:
        如果 return_prompt 为 True:
            Tuple[str, dict, list]: (模型输入提示, 候选实体映射, 原始历史样本列表)
        否则:
            Tuple[List[List[Any]], dict, list]: (历史四元组列表, 候选实体映射, 原始历史样本列表)
    """
    entity, relation, targets, time = x
    
    # 根据参数构建历史图谱
    history_graph = construct_history_by_search(
        entity_search_space, entity, relation, args.history_type
    )

    # 格式化历史记录，获取提示字符串、候选实体和原始历史样本
    history_prompt, candidates, historical_samples = format_history(
        history_graph, args.history_len, [time, entity, relation, targets], args, return_prompt
    )

    # 如果不返回提示字符串，则直接返回历史四元组列表
    if not return_prompt:
        return history_prompt, candidates, historical_samples

    # 构建当前查询的提示部分
    # 格式: "时间戳:[查询实体,查询关系,?]" 或 "[查询实体,查询关系,?]"
    current_prompt = ""
    if not args.no_time:
        current_prompt += f"{time}:"
    current_prompt += f"[{entity},{relation},?]\n"

    # 组合历史提示和当前查询提示
    model_input = history_prompt + current_prompt

    return model_input, candidates, historical_samples


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
    if args.verbose:
        print(f'predictions: {example["predictions"]}')
    for target in example["targets"]:
        metric.total += 1
        index = example["predictions"].index(target) if target in example["predictions"] else -1
        if index >= 0:
            _predictions = [
                x for x in example["predictions"][:index] if x not in example["targets"]
            ]
            rank = len(_predictions) + 1
            if args.verbose:
                print(f"target: {target} --> rank: {rank}")
            metric.update(rank)


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
