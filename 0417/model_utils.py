import json
import math

import numpy as np
import torch


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


def predict(tokenizer, model, prompt, args):
    tokenizer.pad_token_id = tokenizer.eos_token_id # 将结束标记设置为填充标记
    # 1. 将提示输入到模型
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 2. 生成预测结果
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_length, # 最大生成token数
        return_dict_in_generate=True, # 返回一个包含生成结果的词典
        output_scores=True, # 输出每个token的logits
        renormalize_logits=True, # 重新归一化logits
    )
    #     GenerationOutput(
    #     sequences=tensor([[  50,  861, 4965, ...]], device='cuda:0'),
    #     scores=[tensor([[-15.5880, -15.1489, -21.5844, ...]], device='cuda:0'), ...],
    #     hidden_states=[
    #         # 每一层的隐藏状态
    #         (tensor([[[ 0.0172, -0.0145, ..., -0.0457]]], device='cuda:0'), ...),
    #         ...
    #     ]
    # )

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
    return parsed_results # 解析后的结果格式：[(text, probability), ...]——实体id, 概率
