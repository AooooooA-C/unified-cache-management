from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

if hasattr(torch, "npu") and torch.npu.is_available():
    import torch_npu
    import ucm_custom_ops
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

from vllm import _custom_ops as ops
from vllm.attention.ops.flashmla import get_mla_metadata
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from ucm.logger import init_logger
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseRole,
)

if hasattr(torch, "cuda") and torch.cuda.is_available():
    from ucm.sparse.gsa_on_device.hamming_topk import (
        cuda_hamming_topk,
        fake_hamming_topk,
    )
    from ucm.sparse.gsa_on_device.hash_encoder import reshape_and_cache_khash_triton

from ucm.sparse.gsa_on_device.gsa_on_device_config import GSAOnDeviceConfig
from ucm.sparse.gsa_on_device.hash_encoder import HashEncoder
from ucm.utils import Config

logger = init_logger(__name__)

ReqType = Union[str, int]


def gsa_on_device_config_path_for_model(vllm_config) -> str:
    model = vllm_config.model_config.model.lower()
    logger.info("[GSAOnDevice] model name: %s", model)

    if "deepseek" in model and "r1" in model:
        rel = (
            "ucm/sparse/gsa_on_device/configs/gsa_on_device_deepseek_r1_awq_config.json"
        )
    elif "qwen3" in model and "32b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_32B_config.json"
    elif "qwen3" in model and "4b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_4B_config.json"
    elif "qwq" in model and "32b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwq_32B_config.json"
    elif "deepseek" in model and "v2" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_deepseek_v2_lite_config.json"
    elif "qwen3" in model and "30b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_coder_30B_A3B_Instruct_FP8.json"
    else:
        raise ValueError(f"[GSAOnDevice] Unsupported model for gsa_on_device: {model}")

    logger.info("[GSAOnDevice] target relative path: %s", rel)

    cur = Path(__file__).resolve()
    repo = cur
    for depth in range(30):
        if (
            (repo / "pyproject.toml").is_file()
            or (repo / "setup.cfg").is_file()
            or (repo / ".git").exists()
        ):

            p = repo / rel
            logger.info("[GSAOnDevice] repo root detected at depth=%d: %s", depth, repo)
            if p.is_file():
                logger.info("[GSAOnDevice] config loaded from SOURCE tree: %s", p)
                return str(p)
            logger.warning("[GSAOnDevice] repo root found but config missing: %s", p)
            break
        if repo.parent == repo:
            logger.debug("[GSAOnDevice] reached filesystem root, stop searching")
            break

        repo = repo.parent

    sub = rel[len("ucm/") :] if rel.startswith("ucm/") else rel
    res = resources.files("ucm").joinpath(*sub.split("/"))

    with resources.as_file(res) as p:
        logger.info("[GSAOnDevice] config loaded from PACKAGE resource (wheel): %s", p)
        return str(p)


class GSAOnDevice(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.rank = vllm_config.parallel_config.rank
        self.is_mla = vllm_config.model_config.is_deepseek_mla

        if vllm_config.device_config.device_type == "cuda":
            self.is_cuda = True
            self.device = torch.device(f"cuda:{self.rank}")
        elif vllm_config.device_config.device_type == "npu":
            self.is_cuda = False
            self.device = torch.device(f"npu:{self.rank}")
        else:
            raise ValueError(
                f"Unsupported device type: {vllm_config.device_config.device_type}"
            )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.block_size = vllm_config.cache_config.block_size

        # auto detect config file for GSAOnDevice
        gsa_on_device_config_path = gsa_on_device_config_path_for_model(vllm_config)

        self.gsa_on_device_config = GSAOnDeviceConfig.from_json(
            gsa_on_device_config_path
        )
        logger.info(f"read gsa_on_device config file : {gsa_on_device_config_path} ")
        self.hash_topk_tokens = self.gsa_on_device_config.vllm_hash_attention_topk
        self.hash_rollback_layers = (
            self.gsa_on_device_config.vllm_hash_attention_rollback_layers
        )
        self.hash_skip_layers = (
            self.gsa_on_device_config.vllm_hash_attention_skip_layers
        )

        self.seq_len_threshhold = self.gsa_on_device_config.seq_len_threshhold

        if role == UcmSparseRole.WORKER:
            if self.is_cuda:  # cuda only variables
                device_properties = torch.cuda.get_device_properties(self.device)
                num_sms = device_properties.multi_processor_count

                if not vllm_config.model_config.enforce_eager:
                    self.cg_buf_topk_tile_scheduler_metadata = torch.zeros(
                        (num_sms, 8),
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.cg_buf_topk_num_splits = torch.empty(
                        (vllm_config.scheduler_config.max_num_seqs + 1),
                        device=self.device,
                        dtype=torch.int32,
                    )

            self.ori_seq_lens_decode = None
            self.ori_block_table_decode = None
            self.origin_tile_scheduler_metadata = None  # for MLA
            self.origin_num_splits = None  # for MLA

            # for GQA
            self.topk_block_table = None
            self.topk_seq_lens = None
            self.topk_seq_lens_qwen = None
            self.decode_mask = None

            self._k_scale = torch.tensor(1.0, dtype=torch.float32)

            if self.is_mla:
                logger.info("GSAOnDevice initialized with MLA model config")
                self.hash_reduction_head_num = (
                    self.gsa_on_device_config.vllm_hash_attention_reduction_head_num
                )
                self.kv_lora_rank = getattr(
                    vllm_config.model_config.hf_text_config, "kv_lora_rank", None
                )
                self.qk_rope_head_dim = getattr(
                    vllm_config.model_config.hf_text_config, "qk_rope_head_dim", None
                )
                self.hash_encoder_nope = HashEncoder(
                    input_dim=self.kv_lora_rank,
                    hash_bits=self.kv_lora_rank,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )

                self.hash_encoder_rope = HashEncoder(
                    input_dim=self.qk_rope_head_dim,
                    hash_bits=self.qk_rope_head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )
            else:
                logger.info("GSAOnDevice initialized with non-MLA model config")
                self.head_dim = vllm_config.model_config.get_head_size()
                self.hash_encoder = HashEncoder(
                    input_dim=self.head_dim,
                    hash_bits=self.head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )

                self.is_tensor_computed = False
                self.max_batch_size = vllm_config.scheduler_config.max_num_seqs

                if self.is_cuda:  # CUDA only variables
                    self.seq_len_decode = torch.zeros(
                        [self.max_batch_size], dtype=torch.int32, device=self.device
                    )

                else:  # NPU only variables
                    self.decode_mask_npu = None
                    self.hamming_keep_chunks_head = 1
                    self.hamming_keep_chunks_tail = 4

                    self.chunk_sizes_for_hamming_full = torch.full(
                        [self.max_batch_size],
                        fill_value=self.block_size,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.topk_for_hamming_full = torch.full(
                        [self.max_batch_size],
                        fill_value=self.hash_topk_tokens // self.block_size,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.topk_for_hamming_full_cpu = torch.full(
                        [self.max_batch_size],
                        fill_value=self.hash_topk_tokens // self.block_size,
                        dtype=torch.int32,
                        device="cpu",
                    )
                    self.seq_lens_for_hamming = torch.zeros(
                        [self.max_batch_size], dtype=torch.int32, device=self.device
                    )
                    self.hamming_output = torch.zeros(
                        [
                            self.max_batch_size,
                            self.num_key_heads,
                            self.hash_topk_tokens // self.block_size,
                        ],
                        dtype=torch.int32,
                        device=self.device,
                    )

    def hash_code(
        self,
        nope: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None,
        reduction_head_num: int = 1,
        query: Optional[torch.Tensor] = None,
    ):
        if self.is_mla:
            if nope is None or rope is None:
                raise ValueError("MLA mode requires `nope` and `rope`.")
            if reduction_head_num > 1:
                # reduce heads: [T, H, D] -> [T, H/reduce, D]
                nope = nope.view(
                    nope.shape[0],
                    reduction_head_num,
                    nope.shape[1] // reduction_head_num,
                    nope.shape[2],
                ).mean(dim=1)
                rope = rope.view(
                    rope.shape[0],
                    reduction_head_num,
                    rope.shape[1] // reduction_head_num,
                    rope.shape[2],
                ).mean(dim=1)
            hash_nope = self.hash_encoder_nope.compute_hash(nope)
            hash_rope = self.hash_encoder_rope.compute_hash(rope)
            return hash_nope.view(torch.bfloat16), hash_rope.view(torch.bfloat16)

        # ---- GQA mode ----
        else:
            if query is None:
                raise ValueError("GQA mode requires `query`.")
            if self.num_q_heads > self.num_key_heads:
                query = query.view(
                    query.shape[0],
                    self.num_key_heads,
                    self.num_q_heads // self.num_key_heads,
                    query.shape[2],
                ).mean(2)
            elif self.num_q_heads < self.num_key_heads:
                query = torch.repeat_interleave(
                    query, self.num_key_heads // self.num_q_heads, dim=1
                )

            return self.hash_encoder.compute_hash(query).view(torch.bfloat16)

    def get_layer_attn_metadata(self, forward_context: ForwardContext, layer_name: str):
        attn_meta = forward_context.attn_metadata
        attn = forward_context.no_compile_layers[layer_name]
        kv_cache, _ = attn.kv_cache[forward_context.virtual_engine]
        return attn_meta[layer_name] if isinstance(attn_meta, dict) else attn_meta, kv_cache[0]

    def get_layer_state(self, layer_name: str):
        layer_id = int(layer_name.split(".")[2])
        is_rollback_layer = layer_id in self.hash_rollback_layers
        is_skip_hash_layer = (
            layer_id < len(self.hash_skip_layers) and self.hash_skip_layers[layer_id]
        )
        return is_rollback_layer, is_skip_hash_layer
    
    # def get_layer_key_slot_mapping(self, attn_metadata, block_size):
    #     """
    #     修复多卡场景下block_id维度错误，返回CUDA设备上的slot index张量
    #     :param attn_metadata: 包含block_table和seq_lens的元数据对象
    #     :param block_size: int，每个block包含的slot数量
    #     :return:
    #         - all_slot_indices: torch.Tensor (CUDA)，形状 [nums of slots]，一维long张量
    #         - block_to_slots: dict，{block_id: [slot1, slot2,...]}，保留映射关系
    #     """
    #     # 1. 计算有效block数量（确保num_blocks是整数，避免张量类型）
    #     #attn_metadata.seq_lens.shape 是 torch.Size([10]) 的大小是batch_szie
    #     num_blocks = attn_metadata.seq_lens // self.block_size
    #     num_blocks = num_blocks.item() if isinstance(num_blocks, torch.Tensor) else num_blocks

    #     # 2. 提取block id并确保是一维标量张量（核心修复）
    #     # 先切片，再彻底展平为一维，避免残留维度
    #     # attn_metadata.block_table.shape 是torch.Size([10, 256]) 这里10 是 batch_szie
    #     vllm_block_ids = attn_metadata.block_table[:, :num_blocks].flatten()  # 替代squeeze，强制一维
    #     # 确保在CUDA上（和后续张量设备对齐）
    #     if vllm_block_ids.device.type != "cuda":
    #         vllm_block_ids = vllm_block_ids.cuda()

    #     all_slot_indices = []
    #     block_to_slots = {}  

    #     # 3. 遍历block id（确保每个block_id是标量）
    #     for idx in range(len(vllm_block_ids)):
    #         # 用索引取值，确保是0维标量张量
    #         block_id = vllm_block_ids[idx]
    #         # 安全转换为整数（兼容标量张量/普通整数）
    #         block_id_val = block_id.item() if isinstance(block_id, torch.Tensor) else block_id

    #         # 计算当前block的所有slot index
    #         offsets = range(block_size)
    #         slot_indices = [block_id_val * block_size + offset for offset in offsets]
    #         all_slot_indices.extend(slot_indices)
    #         block_to_slots[block_id_val] = slot_indices

    #     # 4. 创建CUDA上的一维张量
    #     target_device = torch.cuda.current_device()
    #     all_slot_indices = torch.tensor(
    #         all_slot_indices, 
    #         dtype=torch.long, 
    #         device=target_device
    #     )

    #     return all_slot_indices, block_to_slots

    def get_slot_list(
        self,
        vllm_block_ids: torch.Tensor,  
        req_len: torch.Tensor,       
        block_size: torch.Tensor,    
    ) -> torch.Tensor:
        ##prefix_len = seq_len - q_len
        t = torch.arange(req_len, device=vllm_block_ids.device, dtype=torch.int64)  
        block_pos = t // block_size                                  
        slot_in_block = t % block_size                              

        block_id = vllm_block_ids.index_select(0, block_pos)                     
        global_slot = block_id * block_size + slot_in_block        

        return global_slot



    def get_layer_key_slot_mapping(self, attn_metadata, key_cache, batch_key, block_size):
        """
        支持多batch场景，返回每个batch对应的slot index张量
        :param attn_metadata: 包含block_table和seq_lens的元数据对象
        :param block_size: int，每个block包含的slot数量
        :return:
            - all_slot_indices: List[torch.Tensor] 或 torch.Tensor，每个batch的slot indices
            如果所有batch的slot数量相同，返回2D张量 [batch_size, num_slots_per_batch]
            否则返回列表，每个元素是 [num_slots_i] 的一维张量
            - block_to_slots: dict，{block_id: [slot1, slot2,...]}，保留映射关系
        """
        flattened_key_cache = key_cache.reshape(key_cache.shape[0] * key_cache.shape[1], *key_cache.shape[2:])
        block_table = attn_metadata.block_table  # shape: [batch_size, max_blocks]
        batch_size = block_table.shape[0]
        
        q_lens = attn_metadata.query_start_loc[1:] -  attn_metadata.query_start_loc[:-1]
        decode_mask = q_lens == 1
        prefix_lens  = attn_metadata.seq_lens - q_lens

        
        all_slot_indices_list = []  # 存储每个batch的slot indices
        
        block_to_slots = {}  # 全局block_id到slot indices的映射
        
        # 遍历每个batch，向量化处理每个batch内的blocks
        for batch_idx in range(batch_size):
            start_loc =attn_metadata.query_start_loc[batch_idx]
            end_loc = attn_metadata.query_start_loc[batch_idx+1]
            batch_block_ids = block_table[batch_idx]  # [max_blocks]
            seq_len = attn_metadata.seq_lens[batch_idx]
            key = batch_key[start_loc:end_loc]
            computed_slot_mapping = attn_metadata.slot_mapping[start_loc:end_loc]
            req_slots = self.get_slot_list(batch_block_ids, seq_len, block_size )
            all_slot_indices_list.append(req_slots)




        #     non_zero_mask = batch_block_ids != 0
        #     valid_block_ids = batch_block_ids[non_zero_mask]  # [num_valid_blocks]
            
        #     if valid_block_ids.numel() == 0:
        #         # 如果当前batch没有有效block，创建空张量
        #         all_slot_indices_list.append(
        #             torch.tensor([], dtype=torch.long, device=block_table.device)
        #         )
        #         continue
            
        #     # 向量化计算当前batch所有block的slot indices
        #     num_valid_blocks = valid_block_ids.shape[0]
            
        #     # 扩展valid_block_ids: [num_valid_blocks] -> [num_valid_blocks, block_size]
        #     block_ids_expanded = valid_block_ids.unsqueeze(1)  # [num_valid_blocks, 1]
        #     offsets = torch.arange(block_size, device=block_table.device, dtype=torch.long)  # [block_size]
            
        #     # 向量化计算：每个block_id生成block_size个slot indices
        #     batch_slot_indices = block_ids_expanded * block_size + offsets  # [num_valid_blocks, block_size]
        #     batch_slot_indices = batch_slot_indices.flatten()  # [num_valid_blocks * block_size]
            
        #     all_slot_indices_list.append(batch_slot_indices)
            
        #     # 更新全局block_to_slots字典（避免重复）
        #     valid_block_ids_cpu = valid_block_ids.cpu().numpy()
        #     for i, block_id_val in enumerate(valid_block_ids_cpu):
        #         block_id_int = int(block_id_val)
        #         if block_id_int not in block_to_slots:
        #             start_idx = i * block_size
        #             end_idx = start_idx + block_size
        #             slot_indices = batch_slot_indices[start_idx:end_idx].cpu().tolist()
        #             block_to_slots[block_id_int] = slot_indices
        
        # # 尝试将所有batch的slot indices合并为2D张量（如果长度一致）
        # # 否则返回列表
        # if len(all_slot_indices_list) == 0:
        #     # 所有batch都为空，返回空张量
        #     all_slot_indices = torch.tensor([], dtype=torch.long, device=block_table.device)
        # else:
        #     slot_lengths = [slots.numel() for slots in all_slot_indices_list]
        #     if len(set(slot_lengths)) == 1 and slot_lengths[0] > 0:
        #         # 所有batch的slot数量相同，可以堆叠为2D张量
        #         all_slot_indices = torch.stack(all_slot_indices_list, dim=0)  # [batch_size, num_slots_per_batch]
        #     else:
        #         # 长度不一致，返回列表
        #         all_slot_indices = all_slot_indices_list
        
        return all_slot_indices_list, block_to_slots


    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ):
        attn_metadata, key_cache = self.get_layer_attn_metadata(forward_context, layer_name)
        # if layer_name =='model.layers.6.self_attn.attn':
        #     print(f"layer_name: {layer_name}, key_cache: {key_cache[1][0]}" )
        # TODO: Should mark MTP layer as rollback layer
        is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)

        if not is_rollback_layer and not is_skip_hash_layer:
            if self.is_mla:
                k_c_normed_hash, k_pe_hash = self.hash_code(nope=key, rope=value)
                ops.concat_and_cache_mla(
                    k_c_normed_hash,
                    k_pe_hash.squeeze(1),
                    k_hash,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype="auto",
                    scale=self._k_scale,
                )
            else:  # GQA
                if self.is_cuda:
                    if not self.is_tensor_computed:
                        if self.decode_mask.any():  # with at least one decode request
                            self.decode_req_ids = torch.nonzero(
                                self.decode_mask, as_tuple=False
                            ).flatten()

                            q_start = attn_metadata.query_start_loc

                            self.decode_token_idx = q_start[:-1].index_select(
                                0, self.decode_req_ids
                            )

                            self.block_table_decode = attn_metadata.block_table.index_select(
                                0, self.decode_req_ids
                            )

                            self.seq_len_decode = self.ori_seq_lens_decode.index_select(
                                0, self.decode_req_ids
                            )
                            self.new_block_table = attn_metadata.block_table
                            self.new_seq_lens = attn_metadata.seq_lens
                            self.is_tensor_computed = True


                    

                    ##  attn_metadata.query_start_loc 得到当前 request 的num_scheued_token,  然后得到对应重算的key的值，在decode的时候， 

                    # if layer_name =='model.layers.6.self_attn.attn':
                    #     print(f"layer_name: {layer_name}, key: {key[0]}" )
                   
                    key_cache_slot_mapping, _ = self.get_layer_key_slot_mapping(attn_metadata,key_cache, key, self.block_size)
                    # total_key_slot_mapping = torch.cat([key_slot_mapping, attn_metadata.slot_mapping], dim=0)
                    # flattened_key_cache = key_cache.reshape(key_cache.shape[0] * key_cache.shape[1], *key_cache.shape[2:])
                    # selected_key_tensor = flattened_key_cache[total_key_slot_mapping]
                    # k_hash_compute = self.hash_encoder.compute_hash(selected_key_tensor).view(
                    #     torch.bfloat16
                    # )
                    # valid_k_hash_token = total_key_slot_mapping.flatten().numel()
                    #attn_metadata.seq_lens 来获得当前request的 seq_lens
                    #key的shape 最多只能到max_num_batched_tokens
                    
                    k_hash_compute = self.hash_encoder.compute_hash(key).view(
                        torch.bfloat16
                    )
                    valid_k_hash_token = attn_metadata.slot_mapping.flatten().numel()

                    ## 这里需要将batch中 request在prefill 阶段，pc 命中时，将pc命中的key + 重算的 key 用于k_hash_compute
                    #所以输入进去当前的attn_metadata 以及key ，返回重新组织的key 和 slot_mapping
                    # attn_metadata.query_start_loc 判断 sheculed tokens,
                    # seq_len query_start_loc -> q_len 比较q_len seq_len - q_len = prefix_len -> > 0 

                    reshape_and_cache_khash_triton(
                        k_hash_compute[:valid_k_hash_token],
                        attn_metadata.slot_mapping.flatten(),
                        k_hash,
                        block_size=self.block_size,
                    )
                else:  # NPU
                    if not self.is_tensor_computed:
                        if self.decode_mask.any():  # with at least one decode request
                            decode_req_ids = torch.nonzero(
                                self.decode_mask, as_tuple=False
                            ).flatten()
                            decode_req_ids_npu = torch.nonzero(
                                self.decode_mask_npu, as_tuple=False
                            ).flatten()
                            batch_size_for_hamming = self.decode_mask.sum().item()
                            self.query_lens_device = attn_metadata.query_lens_device[
                                decode_req_ids_npu
                            ]
                            self.topk_for_hamming = self.topk_for_hamming_full[
                                :batch_size_for_hamming
                            ]
                            self.chunk_sizes_for_hamming = (
                                self.chunk_sizes_for_hamming_full[
                                    :batch_size_for_hamming
                                ]
                            )
                            self.seq_lens_for_hamming = attn_metadata.seq_lens_device[
                                decode_req_ids_npu
                            ]
                            self.max_seq_len_for_hamming = torch.max(
                                attn_metadata.seq_lens[decode_req_ids]
                            ).item()
                            self.is_tensor_computed = True

                    k_hash_compute = self.hash_encoder.compute_hash(key)
                    assert (
                        k_hash_compute.shape[0] == attn_metadata.slot_mapping.numel()
                    ), f"shape mismatch: k_hash_compute.shape[0]={k_hash_compute.shape[0]} != attn_metadata.slot_mapping.numel()={attn_metadata.slot_mapping.numel()}"
                    k_hash_compute = (
                        k_hash_compute.transpose(0, 1)
                        .reshape(-1, k_hash_compute.shape[-1])
                        .contiguous()
                    )
                    ucm_custom_ops.reshape_and_cache_bnsd(
                        k_hash_compute,
                        k_hash,
                        attn_metadata.slot_mapping,
                        attn_metadata.query_lens_device,  # need to modify attention_v1.py in vllm-asecnd
                        k_hash,
                    )
        if self.is_mla:
            if phase == "decode":
                if not is_rollback_layer:
                    if is_skip_hash_layer:
                        assert attn_metadata.decode.topk_block_table is not None
                        block_table = attn_metadata.decode.topk_block_table
                    else:
                        q_nope_hash, q_rope_hash = self.hash_code(
                            nope=decode_ql_nope,
                            rope=decode_q_pe,
                            reduction_head_num=self.hash_reduction_head_num,
                        )
                        q_hash = torch.cat([q_nope_hash, q_rope_hash], dim=-1)
                        topk_token = self.hash_topk_tokens
                        block_table = cuda_hamming_topk(
                            q_hash.unsqueeze(1),
                            k_hash.unsqueeze(2),
                            attn_metadata.decode.block_table,
                            attn_metadata.decode.seq_lens,
                            topk_token=topk_token,
                            sink_token=64,
                            recent_token=512,
                            is_mla=self.is_mla,
                        )
                        attn_metadata.decode.topk_block_table = block_table

                    seq_lens = attn_metadata.decode.topk_seq_lens
                    tile_scheduler_metadata = (
                        attn_metadata.decode.topk_tile_scheduler_metadata
                    )
                    num_splits = attn_metadata.decode.topk_num_splits

                    self.ori_block_table_decode = attn_metadata.decode.block_table
                    self.ori_seq_lens_decode = attn_metadata.decode.seq_lens
                    self.origin_tile_scheduler_metadata = (
                        attn_metadata.decode.tile_scheduler_metadata
                    )
                    self.origin_num_splits = attn_metadata.decode.num_splits

                    attn_metadata.decode.block_table = block_table
                    attn_metadata.decode.seq_lens = seq_lens
                    attn_metadata.decode.tile_scheduler_metadata = (
                        tile_scheduler_metadata
                    )
                    attn_metadata.decode.num_splits = num_splits
        else:  # GQA
            if self.decode_mask.any():  # 有decode阶段的req
                if not is_rollback_layer:
                    if is_skip_hash_layer:
                        # 跳层 使用上一个topk结果
                        attn_metadata.block_table = self.topk_block_table
                        attn_metadata.seq_lens = self.topk_seq_lens
                    else:
                        if self.is_cuda:
                            q_decode = query.index_select(0, self.decode_token_idx)
                            q_hash = self.hash_code(query=q_decode)
                            block_table_decode = cuda_hamming_topk(
                                q_hash.unsqueeze(1),
                                k_hash,
                                self.block_table_decode,
                                self.seq_len_decode,
                                topk_token=self.hash_topk_tokens,
                                sink_token=64,
                                recent_token=512,
                                is_mla=self.is_mla,
                            )
                            # update topk_block_table
                            topk = block_table_decode.shape[1]
                            self.new_block_table[self.decode_req_ids, :topk] = (
                                block_table_decode
                            )
                            self.new_block_table[self.decode_req_ids, topk:] = 0
                            attn_metadata.block_table = self.new_block_table
                            self.new_seq_lens[self.decode_mask] = self.topk_seq_lens_qwen
                            attn_metadata.seq_lens = self.new_seq_lens
                            
                        else:  # NPU

                            decode_req_ids = torch.nonzero(
                                self.decode_mask_npu, as_tuple=False
                            ).flatten()
                            decode_token_idx = q_start[:-1].index_select(
                                0, decode_req_ids
                            )
                            q_decode = query.index_select(0, decode_token_idx)

                            q_hash = (
                                self.hash_encoder.compute_hash(q_decode)
                                .unsqueeze(2)
                                .contiguous()
                            )

                            block_table_decode = attn_metadata.block_table.index_select(
                                0, decode_req_ids
                            )

                            ucm_custom_ops.hamming_dist_top_k(
                                q_hash,
                                k_hash,
                                self.topk_for_hamming,
                                self.seq_lens_for_hamming,
                                self.chunk_sizes_for_hamming,
                                self.max_seq_len_for_hamming,
                                self.hamming_keep_chunks_head,
                                self.hamming_keep_chunks_tail,
                                0,  # support_offload is disabled
                                block_table_decode,
                                self.hamming_output[: len(decode_req_ids)],
                            )
                            topk = self.hamming_output.shape[-1]
                            attn_metadata.block_table[decode_req_ids, :topk] = (
                                self.hamming_output[: len(decode_req_ids), 0, :]
                            )
                            attn_metadata.block_table[decode_req_ids, topk:] = 0

                            # we have already computed the topk_seq_lens_qwen in `build_decode_attention_meta_npu()`
                            attn_metadata.seq_lens[self.decode_mask] = (
                                self.topk_seq_lens_qwen
                            )

                        # topk for skip layer
                        self.topk_block_table = attn_metadata.block_table
                        self.topk_seq_lens = attn_metadata.seq_lens

        return query, key, value, output

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        attn_metadata,_ = self.get_layer_attn_metadata(forward_context, layer_name)
        if self.is_mla:
            if phase == "decode":
                # TODO: Should mark MTP layer as rollback layer
                is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
                if not is_rollback_layer:
                    attn_metadata.decode.block_table = self.ori_block_table_decode
                    attn_metadata.decode.seq_lens = self.ori_seq_lens_decode
                    attn_metadata.decode.tile_scheduler_metadata = (
                        self.origin_tile_scheduler_metadata
                    )
                    attn_metadata.decode.num_splits = self.origin_num_splits
        else:  # 判断req decode阶段
            if self.decode_mask.any():
                attn_metadata.block_table = self.ori_block_table_decode
                attn_metadata.seq_lens = self.ori_seq_lens_decode
            # else:
            #     self.full_key_slotmapping = None

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def execute_begin(self, scheduler_output: SchedulerOutput):
        self.is_tensor_computed = False

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        return INVALID_SLOT

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        dtype = torch.bfloat16
        for layer_name, kv_cache in kv_caches.items():
            khash_cache_shape = list((kv_cache if self.is_mla else kv_cache[0]).shape)
            khash_cache_shape[-1] //= dtype.itemsize * 8
            khash_cache = torch.zeros(khash_cache_shape, dtype=dtype, device=device)
            kv_caches[layer_name] = (kv_cache, khash_cache)

    def initialize_kv_hash_cache_tensors_npu(self, kv_caches, device):
        print(
            f"[NPU GSAOnDevice Debug] initialize_kv_hash_cache_tensors_npu: allocating hashk cache for GSAOnDevice in NPU"
        )
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            k_cache_shape = kv_cache[0].shape
            print(
                f"[NPU GSAOnDevice Debug] layer_name: {layer_name}, is_rollback_layer={is_rollback_layer}, is_skip_hash_layer={is_skip_hash_layer}, k_cache_shape: {k_cache_shape}"
            )
            khash_cache_shape = (
                k_cache_shape[0],
                k_cache_shape[2],
                k_cache_shape[1],
                self.hash_encoder.hash_bits // 8,
            )
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_cache = torch.empty(
                    khash_cache_shape, dtype=torch.uint8, device=device
                )
                print(
                    f"[NPU GSAOnDevice Debug] layer_name: {layer_name}, khash_cache_shape: {khash_cache_shape}"
                )
            else:
                khash_cache = None
                print(
                    f"[NPU GSAOnDevice Debug] layer_name: {layer_name}, khash_cache is None"
                )
            kv_caches[layer_name] = (kv_cache, khash_cache)

    def build_decode_hash(self, seq_lens):
        from ucm.sparse.gsa_on_device.hamming_topk import update_seq_lens

        topk_seq_lens = update_seq_lens(
            seq_lens,
            topk_token=self.hash_topk_tokens,
            block_size=self.block_size,
        )
        topk_tile_scheduler_metadata, topk_num_splits = get_mla_metadata(
            topk_seq_lens,
            self.num_q_heads,
            1,
        )
        return topk_seq_lens, topk_tile_scheduler_metadata, topk_num_splits

    def build_decode_attention_meta(self, query_start_loc, seq_lens, block_table):

        from ucm.sparse.gsa_on_device.hamming_topk import update_seq_lens

        q_lens = query_start_loc[1:] - query_start_loc[:-1]
        self.decode_mask = q_lens == 1

        self.ori_seq_lens_decode = seq_lens.clone()
        self.ori_block_table_decode = block_table.clone()
        if self.decode_mask.any():
            decode_seq_lens = seq_lens[self.decode_mask]
            self.topk_seq_lens_qwen = update_seq_lens(
                decode_seq_lens,
                topk_token=self.hash_topk_tokens,
                block_size=self.block_size,
            )
        return self.decode_mask, self.topk_seq_lens_qwen

    def build_decode_attention_meta_npu(self, query_lens, seq_lens, block_table):

        from ucm.sparse.gsa_on_device.hamming_topk import update_seq_lens

        # self.decode_mask is on cpu in vllm-asencd under NPU device
        self.decode_mask = (query_lens == 1) & (seq_lens >= self.seq_len_threshhold)
        self.decode_mask = self.decode_mask.pin_memory()

        self.ori_seq_lens_decode = seq_lens.clone()
        self.ori_block_table_decode = block_table.clone()

        self.decode_mask_npu = self.decode_mask.to(self.device, non_blocking=True)

        if self.decode_mask.any():
            decode_seq_lens = seq_lens[self.decode_mask]
            self.topk_seq_lens_qwen = update_seq_lens(
                decode_seq_lens,
                topk_token=self.hash_topk_tokens,
                block_size=self.block_size,
            )

    def maybe_init_cudagraph_buffers_for_topk(self, n, tile_scheduler_metadata):
        sm_parts = tile_scheduler_metadata.size(0)
        topk_tile_scheduler_metadata_view = self.cg_buf_topk_tile_scheduler_metadata[
            :sm_parts
        ]
        topk_tile_scheduler_metadata_view.copy_(topk_tile_scheduler_metadata)
        topk_tile_scheduler_metadata = topk_tile_scheduler_metadata_view

        topk_num_splits_view = self.cg_buf_topk_num_splits[:n]
        topk_num_splits_view.copy_(topk_num_splits)
        self.cg_buf_topk_num_splits[n:].fill_(topk_num_splits[-1])
        topk_num_splits = topk_num_splits_view
        return topk_tile_scheduler_metadata, topk_num_splits
