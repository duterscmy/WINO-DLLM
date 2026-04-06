import torch
import torch.nn.functional as F
import numpy as np
import time

@ torch.no_grad()
def decoding_default(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Default decoding function from LLaDA paper
    '''
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    # attention_mask: bs, 1, seq_len, seq_len, all true
    attention_mask = torch.ones(1, 1, x.shape[1], x.shape[1], dtype=torch.bool).to(model.device)
    
    position_ids = torch.arange(x.shape[1], device=x.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask, position_ids=position_ids).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x, steps * num_blocks

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
        
     
@torch.no_grad()
def decoding_wino(model, prompt, gen_length=128, block_length=128, temperature=0., mask_id=126336, threshold=0.6, threshold_back=0.9):

    device = model.device
    x_block = torch.full((1, prompt.shape[1] + gen_length + block_length), mask_id, dtype=torch.long).to(model.device)
    x_block[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x_block != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    step = 0
    

    for num_block in range(num_blocks):
        block_step = 0
        mask_index_block = (x_block == mask_id) # b, l
        mask_index_block[:, prompt.shape[1] + (num_block + 1) * block_length:] = False
        
        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:,  -block_length:] = ~mask_index_block[:, prompt.shape[1] + num_block* block_length: prompt.shape[1] + (num_block + 1) * block_length]
        position_ids = torch.cat([torch.arange(prompt.shape[1] + gen_length, device=device), torch.arange(prompt.shape[1] + num_block * block_length, prompt.shape[1] + (num_block + 1) * block_length, device=device)])
        attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = torch.ones(block_length, block_length, dtype=torch.bool).to(device)
        attention_mask[:, :, -block_length:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = ~torch.eye(block_length, dtype=torch.bool).to(device)
        last_accept = 30
        while mask_index_block.any():
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits # b, l, vocab_size
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_block_shift_left[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_block_shift_left]

            p = F.softmax(logits.to(torch.float64), dim=-1) # b, l, vocab_size
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            x0 = torch.where(mask_index_block, x0, x_block) # replace the masked tokens with the predicted tokens
            confidence = torch.where(mask_index_block, x0_p, -np.inf) # keep the confidence of the masked tokens
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)
            

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                # get top max_accept tokens
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            
            # always transfer the max confidence token
            else:
                if not transfer_index.any():
                    max_confidence_index = torch.argmax(confidence)
                    transfer_index.view(-1)[max_confidence_index] = True
            x_block[transfer_index] = x0[transfer_index]
            
            num_accept = transfer_index.sum()
            
            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    confidence_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                    _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)
            
            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            block_step += 1
            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
            unmask_index_block[transfer_index_shift] = True
            unmask_index_block[remask_index] = False
            last_accept = num_accept

        step += block_step

    return x_block[:, :prompt.shape[1] + gen_length], step


@torch.no_grad()
def decoding_soar(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                        cfg_scale=0., remasking='low_confidence', mask_id=126336, max_beam_size=2, log=False, logits_eos_inf=False, confidence_eos_eot_inf=False, **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        max_beam_size: Maximum beam size for dynamic beam search.
    '''
    # temperature=0.0
    print("======soar, temperature: {:.1f}====".format(temperature))
    import json
    
    # 配置参数
    confidence_threshold = 0.80  # 高置信度阈值
    min_parallel_tokens = 1      # 并行解码最小token数
    max_parallel_tokens = 5      # 并行解码最大token数
    
    # 初始化beam: [(sequence, cumulative_log_prob, block_progress, records)]
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # 每个beam有自己的records列表
    beam = [(x.clone(), 0.0, 0, [])]  # (sequence, cumulative_log_prob, current_block, records)
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    if log:
        print(f"=== Dynamic Beam Search Generation Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}, Max beam size: {max_beam_size}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Initial mask count: {(x == mask_id).sum().item()}")
        
    step = 0
    for global_step in range(steps):
        step += 1
        if log:
            print(f"=== Global Step {global_step + 1}/{steps} ===")
        
        # 检查所有beam中的序列是否还有mask
        has_remaining_masks = False
        for seq, _, _, _ in beam:
            if (seq == mask_id).any():
                has_remaining_masks = True
                break
        
        # 如果没有mask了，提前结束
        if not has_remaining_masks:
            print(f"No masks remaining in any beam, early stopping at step {global_step + 1}")
            break
        
        # 批量收集所有beam中的序列
        beam_sequences = [seq for seq, _, _, _ in beam]
        batch_sequences = torch.cat(beam_sequences, dim=0)  # (beam_size, seq_len)
        
        if log:
            print(f"Processing batch of {len(beam)} sequences")
            total_masks = sum([(seq == mask_id).sum().item() for seq in beam_sequences])
            print(f"Total remaining masks across all beams: {total_masks}")
        
        # 批量计算所有序列的logits
        with torch.no_grad():
            if cfg_scale > 0.:
                # 对每个序列都需要unconditional版本
                unconditional_seqs = []
                for seq in beam_sequences:
                    un_seq = seq.clone()
                    un_seq[prompt_index] = mask_id
                    unconditional_seqs.append(un_seq)
                
                unconditional_batch = torch.cat(unconditional_seqs, dim=0)
                combined_batch = torch.cat([batch_sequences, unconditional_batch], dim=0)
                
                batch_logits = model(combined_batch).logits
                conditional_logits, unconditional_logits = torch.chunk(batch_logits, 2, dim=0)
                batch_logits = unconditional_logits + (cfg_scale + 1) * (conditional_logits - unconditional_logits)
            else:
                batch_logits = model(batch_sequences).logits
        
        if logits_eos_inf:
            batch_logits[:, :, 126081] = -torch.inf
        # 为每个序列添加Gumbel噪声并获取预测
        logits_with_noise = add_gumbel_noise(batch_logits, temperature=temperature)
        batch_x0 = torch.argmax(logits_with_noise, dim=-1)
        
        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = batch_logits[:, :, 126348] = -torch.inf
        
        if remasking == 'low_confidence':
            p = F.softmax(batch_logits, dim=-1)
            batch_x0_p = torch.gather(p, dim=-1, index=batch_x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == 'random':
            batch_x0_p = torch.rand(batch_x0.shape, device=batch_x0.device)
        else:
            raise NotImplementedError(remasking)
        
        new_beam_candidates = []
        has_multi_unmask_candidate = False
        
        # 并行处理每个序列的候选生成
        for beam_idx, (seq, cumulative_log_prob, current_block, records) in enumerate(beam):
            logits = batch_logits[beam_idx:beam_idx+1]
            x0 = batch_x0[beam_idx:beam_idx+1]
            x0_p = batch_x0_p[beam_idx:beam_idx+1]

            if log:
                print(f"--- Processing Beam {beam_idx + 1}/{len(beam)} ---")
                print(f"Current cumulative log prob: {cumulative_log_prob:.4f}")
                print(f"Current block progress: {current_block}/{num_blocks}")
            
            # 检查当前序列是否还有mask
            if not (seq == mask_id).any():
                new_beam_candidates.append((seq, cumulative_log_prob, current_block, records))
                if log:
                    print(f"    Sequence already complete (no masks remaining)")
                continue
            
            # 确定当前处理的block
            block_start = prompt.shape[1] + current_block * block_length
            block_end = prompt.shape[1] + (current_block + 1) * block_length
            
            # 获取当前block的mask信息
            mask_index = (seq == mask_id)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # 限制在当前block及之前
            confidence[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf
            
            # 获取当前block内的mask位置和置信度
            block_mask_positions = torch.where(mask_index[0, block_start:block_end])[0] + block_start
            block_mask_confidence = confidence[0, block_mask_positions]
            
            if log:
                print(f"Block {current_block} mask positions: {block_mask_positions.cpu().numpy().tolist()}")
                print(f"Block {current_block} mask confidences: {block_mask_confidence.cpu().float().detach().numpy().tolist()}")
            
            # 策略选择：检查是否有足够的高置信度token进行并行解码
            high_confidence_mask = block_mask_confidence >= confidence_threshold
            high_confidence_indices = torch.where(high_confidence_mask)[0]
            
            if len(high_confidence_indices) >= min_parallel_tokens:
                # 策略(1): 并行解码多个高置信度token
                if log:
                    print(f"Strategy 1: Parallel decoding {len(high_confidence_indices)} high-confidence tokens")
                
                # 选择置信度最高的前几个token进行并行解码
                num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)
                top_probs, top_indices = torch.topk(block_mask_confidence[high_confidence_indices], num_to_unmask)
                selected_indices = high_confidence_indices[top_indices]
                
                new_seq = seq.clone()
                new_log_prob = cumulative_log_prob
                new_records = records.copy()  # 复制之前的records
                
                # 一次性解码多个token
                for idx in range(num_to_unmask):
                    original_idx = selected_indices[idx].item()
                    pos = block_mask_positions[original_idx].item()
                    token = x0[0, pos].item()
                    prob = top_probs[idx].item()
                    
                    new_seq[0, pos] = token
                    new_log_prob += prob
                    
                    # 为每个并行解码的token添加记录
                    new_records.append({
                        "step": global_step + 1,
                        "position": pos,
                        "confidence": prob,
                        "token_id": token
                    })
                
                # 更新block进度
                new_current_block = current_block
                if new_current_block < num_blocks - 1:
                    current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                    if not current_block_mask.any():
                        new_current_block += 1
                        if log:
                            print(f"    Block {current_block} completed after parallel decoding, moving to block {new_current_block}")
                
                new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
                has_multi_unmask_candidate = True
                
                if log:
                    print(f"    Parallel unmasked {num_to_unmask} tokens")
                    print(f"    New cumulative log prob: {new_log_prob:.4f}")
                
            else:
                # 策略(2): Beam search探索 - 选择top k个位置逐个解码
                k = min(max_beam_size, len(block_mask_confidence))
                if k == 0:
                    # 当前block没有mask了，移动到下一个block
                    new_current_block = min(current_block + 1, num_blocks - 1)
                    new_beam_candidates.append((seq, cumulative_log_prob, new_current_block, records))
                    if log:
                        print(f"    No masks in current block, moving to block {new_current_block}")
                    continue
                
                top_probs, top_indices = torch.topk(block_mask_confidence, k)
                top_positions = block_mask_positions[top_indices]
                top_tokens = x0[0, top_positions]
                
                if log:
                    print(f"Strategy 2: Beam search with k={k}")
                
                for idx in range(k):
                    new_seq = seq.clone()
                    pos = top_positions[idx].item()
                    token = top_tokens[idx].item()
                    prob = top_probs[idx].item()
                    
                    new_seq[0, pos] = token
                    new_log_prob = cumulative_log_prob + prob
                    
                    # 更新block进度
                    new_current_block = current_block
                    if new_current_block < num_blocks - 1:
                        current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                        if not current_block_mask.any():
                            new_current_block += 1
                    
                    # 创建新的records列表并添加当前解码记录
                    new_records = records.copy()
                    new_records.append({
                        "step": global_step + 1,
                        "position": pos,
                        "confidence": prob,
                        "token_id": token
                    })
                    
                    new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
                    
                    if log and idx < 2:
                        print(f"    Unmask position {pos}, confidence {prob:.4f}")
        
        # 如果没有生成新的候选，提前结束
        if not new_beam_candidates:
            if log:
                print("No new beam candidates generated, early stopping")
            break
        
        # 全局排序，选择top sequences
        if log:
            print(f"Total candidates before selection: {len(new_beam_candidates)}")
        
        # 按log概率排序
        new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 去重
        uniq_new_beam_candidates = []
        seen = set()
        for tensor, log_prob, block_progress, records in new_beam_candidates:
            tensor_tuple = tuple(tensor.flatten().cpu().numpy().tolist())
            if tensor_tuple not in seen:
                seen.add(tensor_tuple)
                uniq_new_beam_candidates.append((tensor, log_prob, block_progress, records))
        
        if log:
            print(f"Unique candidates after deduplication: {len(uniq_new_beam_candidates)}")
        
        # 动态调整beam size
        if has_multi_unmask_candidate and uniq_new_beam_candidates:
            best_candidate = uniq_new_beam_candidates[0]
            best_seq, best_log_prob, best_block, best_records = best_candidate
            
            original_mask_count = (beam[0][0] == mask_id).sum().item()
            current_mask_count = (best_seq == mask_id).sum().item()
            masks_unmasked = original_mask_count - current_mask_count
            
            if masks_unmasked >= min_parallel_tokens:
                # 策略(1): 并行解码了多个token，缩小beam_size=1
                beam_size = 1
                beam = [best_candidate]
                if log:
                    print(f"Dynamic adjustment: Parallel unmasked {masks_unmasked} tokens, beam_size reduced to 1")
            else:
                # 策略(2): 扩大beam_size
                beam_size = min(max_beam_size, len(uniq_new_beam_candidates))
                beam = uniq_new_beam_candidates[:beam_size]
                if log:
                    print(f"Dynamic adjustment: Standard decoding, beam_size set to {beam_size}")
        else:
            # 策略(2): 没有并行解码候选，扩大beam_size
            beam_size = min(max_beam_size, len(uniq_new_beam_candidates))
            beam = uniq_new_beam_candidates[:beam_size]
            if log:
                print(f"Dynamic adjustment: No parallel decoding, beam_size set to {beam_size}")
        
        # 打印当前beam状态
        best_seq, best_score, best_block, best_records = beam[0]
        if log:
            print(f"Current beam size: {len(beam)}")
            print(f"Best sequence score: {best_score:.4f}")
            print(f"Remaining mask count: {(best_seq == mask_id).sum().item()}")
            

    # 选择beam中最好的序列作为最终结果
    if beam:
        best_sequence, best_score, _, best_records = beam[0]
        
        if log:
            print(f"=== Dynamic Beam Search Generation Complete ===")
            print(f"Final sequence score: {best_score:.4f}")
            print(f"Final mask count: {(best_sequence == mask_id).sum().item()}")
            print(f"Total decoding records: {len(best_records)}")
            
            # 输出records的简单统计
            if best_records:
                steps_used = max(r["step"] for r in best_records)
                avg_confidence = sum(r["confidence"] for r in best_records) / len(best_records)
                print(f"Steps used: {steps_used}")
                print(f"Average confidence: {avg_confidence:.4f}")
            
            if not (best_sequence == mask_id).any():
                print(f"✓ All masks have been filled!")
            else:
                print(f"⚠ Still has {(best_sequence == mask_id).sum().item()} masks remaining")
            
            # 输出前几个解码记录作为示例
            print(f"\n=== Top 5 Decoding Records ===")
            for i, record in enumerate(best_records[:5]):
                print(f"Step {record['step']}: position {record['position']}, "
                      f"token {record['token_id']}, confidence {record['confidence']:.4f}")
    else:
        best_sequence = x
        best_records = []
        if log:
            print(f"=== Dynamic Beam Search Generation Complete (No valid sequences) ===")

    # 只输出top1 beam的records作为JSON
    import json
    print(json.dumps(best_records))
    print(len(best_records))
    return best_sequence, step




def decoding_wino_soar_hybrid(model, prompt, steps=128, gen_length=128, block_length=32,
                                  temperature=0., cfg_scale=0., remasking='low_confidence',
                                  mask_id=126336, max_beam_size=2, threshold_back=0.9,
                                  threshold=0.6):
    """
    混合解码策略的简化稳定版本
    """
    device = model.device
    
    # 初始化
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)
    
    total_steps = 0
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        current_prompt = x[:, :block_start]
        
        # 阶段1: SOAR解码
        soar_decoded, soar_steps = decoding_soar(
            model=model, prompt=current_prompt, steps=steps_per_block,
            gen_length=block_length, block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
            max_beam_size=max_beam_size, log=False
        )
        
        x[:, block_start:block_end] = soar_decoded[:, current_prompt.shape[1]:]
        total_steps += soar_steps
        
        # 阶段2: Wino重掩码修正
        for refine_iter in range(5):  # 最多5次修正
            # 计算置信度
            with torch.no_grad():
                logits = model(x).logits
                if cfg_scale > 0:
                    uncond = x.clone()
                    uncond[:, :prompt.shape[1]] = mask_id
                    combined = torch.cat([x, uncond], dim=0)
                    logits_combined = model(combined).logits
                    cond_logits, uncond_logits = torch.chunk(logits_combined, 2, dim=0)
                    logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                
                p = F.softmax(logits.to(torch.float64), dim=-1)
                confidence = torch.gather(p, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
            
            # 检查当前block的低置信度token
            block_confidence = confidence[0, block_start:block_end]
            is_masked = (x[0, block_start:block_end] == mask_id)
            low_conf = (block_confidence < threshold_back) & (~is_masked)
            
            if not low_conf.any():
                break
            
            # 重掩码
            remask_positions = torch.where(low_conf)[0] + block_start
            x[0, remask_positions] = mask_id
            
            # 重新解码这些位置（一次解码一个）
            for _ in range(len(remask_positions)):
                with torch.no_grad():
                    logits = model(x).logits
                    if cfg_scale > 0:
                        uncond = x.clone()
                        uncond[:, :prompt.shape[1]] = mask_id
                        combined = torch.cat([x, uncond], dim=0)
                        logits_combined = model(combined).logits
                        cond_logits, uncond_logits = torch.chunk(logits_combined, 2, dim=0)
                        logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                    
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                
                # 只解码当前block内的mask位置
                block_mask = (x[0, block_start:block_end] == mask_id)
                if block_mask.any():
                    block_mask_positions = torch.where(block_mask)[0] + block_start
                    # 选择置信度最高的位置解码
                    conf_at_masks = x0_p[0, block_mask_positions]
                    best_idx = torch.argmax(conf_at_masks)
                    best_pos = block_mask_positions[best_idx]
                    x[0, best_pos] = x0[0, best_pos]
                    total_steps += 1
            
            total_steps += 1
    
    return x, total_steps

#