import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM


def permute(tokenizer, scores, cur_step, max_step, cur_seq, seqs, dec_cand, end_char):
    if cur_step == max_step or (len(cur_seq) > 0 and end_char in cur_seq[-1]["token"]):
        _cur_seq = cur_seq[:-1].copy() if end_char in cur_seq[-1]["token"] else cur_seq.copy()
        normalized_logit = (
            sum([x["logit"] for x in _cur_seq]) / len(_cur_seq) if len(_cur_seq) > 0 else -math.inf
        )
        seqs.append(
            {
                "tokens": [x["token"] for x in _cur_seq],
                "text": "".join([x["token"] for x in _cur_seq]).strip(),
                "probability": normalized_logit,
            }
        )
        return
    logits = scores[cur_step]
    logits_indices = torch.argsort(logits, dim=-1, descending=True)
    for tok in logits_indices[0][:dec_cand]:
        cur_seq.append({"token": tokenizer.decode(tok), "logit": logits[0][tok].item()})
        permute(tokenizer, scores, cur_step + 1, max_step, cur_seq, seqs, dec_cand, end_char)
        cur_seq.pop()


def deduplicate(x):  # NOTE: assumes a sorted list based on probability
    f = {}
    z = []
    for y in x:
        if y[0] in f:
            continue
        f[y[0]] = True
        z.append(y)
    return z


def parse_results(results):
    '''
    results: [(text, probability), ...]
    text: 实体id
    probability: 概率

    return: [(entity_id, probability), ...]
    entity_id: 实体id
    probability: 概率
    '''
    logprobs = [(int(x["text"]), x["probability"]) for x in results if x["text"].isdecimal()]
    sorted_logprobs = sorted(logprobs, key=lambda tup: tup[1], reverse=True)
    dedup_sorted_logprobs = deduplicate(sorted_logprobs)

    probs = [x[1] for x in dedup_sorted_logprobs]
    softmax_probs = np.exp(probs) / np.sum(np.exp(probs), axis=0)

    to_return = [(x[0], p) for x, p in zip(dedup_sorted_logprobs, softmax_probs)]
    return to_return


def predict(model, tokenizer, prompt, args, return_scores=False):
    """通用预测函数"""

    # --- 移除调用栈打印 ---
    # print("DEBUG PREDICT: predict() function called. Printing stack trace:")
    # traceback.print_stack()
    # print("--- End of stack trace ---")
    # --- 移除结束 ---

    # 确保模型和分词器已加载
    if model is None or tokenizer is None:
        print("Error: Model or tokenizer not provided to predict function.")
        return None

    # 确定设备
    device = model.device

    # --- DEBUG: 打印传入的 prompt ---
    print(f"DEBUG_PREDICT: Received prompt (first 500 chars): {prompt[:500]}...")
    # --- END DEBUG ---
    tokenizer.pad_token_id = tokenizer.eos_token_id # 将结束标记设置为填充标记
    # 1. 将提示输入到模型
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 2. 生成预测结果
    with torch.no_grad(): # 确保在推理模式下运行，不计算梯度
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_length, # 最大生成token数
            return_dict_in_generate=True, # 返回一个包含生成结果的词典
            output_scores=True, # 输出每个token的logits
            renormalize_logits=True, # 重新归一化logits
        )

    # --- DEBUG: 打印模型原始输出 ---
    try:
        # 尝试解码完整的生成序列
        full_generated_sequence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print(f"DEBUG_PREDICT: Raw model output sequences (decoded): {full_generated_sequence}")
        # 打印 scores 的形状（如果存在且非空）
        if hasattr(outputs, 'scores') and outputs.scores:
            print(f"DEBUG_PREDICT: Raw model output scores shape (first score tensor): {outputs.scores[0].shape}")
        else:
            print("DEBUG_PREDICT: Raw model output scores not available or empty.")
    except Exception as e:
        print(f"DEBUG_PREDICT: Error decoding raw output or accessing scores: {e}")
    # --- END DEBUG ---

    if args.verbose: # 是否打印详细信息
        print("outputs:\n")
    if args.label and "llama" not in args.model:
        '''
        # 其他模型（如Qwen、ChatGLM等）可能能很好地处理带标签的格式
        # "[entity,relation,0.target]"  # args.label=True 的格式
        # 而LLaMA可能对这种格式不够友好，更适合直接使用
        # "[entity,relation,target]"    # args.label=False 的格式
        '''
        probs = outputs.scores[0]  # 只取第一个token的logits # (1, num_vocab)
        probs_indices = torch.argsort(probs, dim=-1, descending=True)  # 对logits进行排序，并返回索引 # (1, num_vocab)
        # 3. 解析预测结果
        results = []
        for tok in probs_indices[0][: args.top_k]: # 只取前k个
            if args.verbose:
                print(
                    f"| {tok:5d} | {tokenizer.decode(tok):8s} | {probs[0][tok].item():.4f} | {np.exp(probs[0][tok].item()):.2%}"
                )
            results.append(
                {
                    "text": tokenizer.decode(tok).strip(), # 解码token
                    "probability": probs[0][tok].item(), # 概率
                }
            )
    else:
        results = []
        permute(
            tokenizer,
            outputs.scores,
            0,
            args.max_length,
            [],
            results,
            args.dec_cand,
            "." if args.label and not args.no_entity else "]",
        )
        results = list(sorted(results, key=lambda x: x["probability"], reverse=True))[: args.top_k]
        if args.verbose:
            for x in results:
                print(
                    f'| {json.dumps(x["tokens"]):30s} | {x["text"]:10s} | {x["probability"]:.4f} | {np.exp(x["probability"]):.2%}'
                )

    # 4. 解析预测结果
    parsed_results = parse_results(results)

    # --- DEBUG: 打印最终返回的解析结果 ---
    print(f"DEBUG_PREDICT: Parsed results (first 10): {parsed_results[:10]}")
    # +++ Add logging for the full parsed_results before returning +++
    print(f"DEBUG_PREDICT: Returning parsed_results (len={len(parsed_results)}): {parsed_results}")
    # --- END DEBUG ---

    return parsed_results # 解析后的结果格式：[(text, probability), ...]
