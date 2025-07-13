import torch.nn as nn
from typing import Tuple,List,Callable
from torch.autograd import Variable
import queue
import torch
#### WeightGradStore

class WeightGradStore:
    enabled : bool = False
    cache : List[Callable] = []
    functions_queue = queue.Queue()
    
    @classmethod
    def put(cls,function:Callable) -> None:
        cls.cache.append(function)
        
    @classmethod 
    def flush(cls) -> None:
        cls.functions_queue.put(cls.cache)
        cls.cache = []
        
    @classmethod
    def pop(cls) -> None:
        assert not cls.functions_queue.empty() , "Can't pop empty queue"
        functions = cls.functions_queue.get()
        for function in functions:
            function()
            
    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.functions_queue = queue.Queue()
        
def run_backward(tensors:  List[torch.Tensor],grad_tensors : List[torch.Tensor])-> None:
    
    kwargs = dict(
        keep_graph = False,
        create_graph = False,
        allow_uncreachable = True,
        accumulate_grad = True,
    )

    Variable._execution_engine.run_backward(tensors,grad_tensors,**kwargs)
    
    
def chunk_tensor(x:torch.Tensor,chunks,dim):
    if x is None:
        return [None for _ in range(chunks)]
    
    return x.tensor_split(chunks,dim=dim)

def cat_tensor(x,dim):
    if (isinstance(x,tuple) or isinstance(x,list)):
        if len(x) ==1 :
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x,dim=dim)

def scatter(inputs,chunks,dim):
    assert isinstance(inputs,(torch.Tensor,tuple,list))
    
    if isinstance(inputs,torch.Tensor):
        inputs = (inputs,)

    assert all(x is None or isinstance(x,torch.Tensor) for x in inputs)
    
    inputs = [chunk_tensor(x,chunks,dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
        
    return microbatches

def gather(micro_ouputs,dim):
    assert isinstance(micro_ouputs[0],(torch.Tensor,tuple,list))
    if isinstance(micro_ouputs[0],torch.Tensor):
        micro_ouputs = [(x,) for x in micro_ouputs]
    
    outputs = [x for x in zip(*micro_ouputs)]
    outputs = tuple(cat_tensor(x,dim=dim) for x in outputs)    
    return outputs

#####
## comm functions
#####

from typing import List,Tuple
import torch.distributed as dist

TENSOR_SAHPES : List[Tuple[int]] = None
TENSOR_DTYPE : torch.dtype = None

def set_p2p_tensor_shapes(shapes:List[Tuple[int]]):
    global TENSOR_SAHPES
    TENSOR_SAHPES = shapes

def set_p2p_tensor_dtype(dtype:torch.dtype):
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype
    
def build_from_tensor_shapes():
    return [torch.empty(s,dtype=TENSOR_DTYPE,device="cuda",requires_grad=True) for s in TENSOR_SAHPES]

def append_irecv(ops: List[dist.P2POp],src:int,group:dist.ProcessGroup) -> List[torch.Tensor]:
    tensors = build_from_tensor_shapes()
    src = dist.distributed_c10d.get_global_rank(group,src)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv,tensor,src))
    return tensors

def append_isend(ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int, group: dist.ProcessGroup) -> None:
    dst = dist.distributed_c10d.get_global_rank(group, dst)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))        


####
# DualPipe Class
###
from typing import Tuple, List, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist


class DualPipe(nn.Module):
    def __init__(
        self,
        modules : Tuple[nn.Module,nn.Module],
        batch_dim : int = 0,
        process_group : Optional[dist.ProcessGroup] = None,
        rank_mapping: Optional[List[int]] = None,   
    ) -> None:
        super().__init__()
        
        assert next(modules[0].parameters()).device == torch.device(torch.cuda.current_device())
        self.module = nn.ModuleList(modules)
        
        self.overlapped_forward_backward = type(modules[0]) == type(modules[1]) and hasattr(type[modules[0]],"overlapped_forward_backwarrd")
        
        self.batch_dim = batch_dim
        self.group = process_group or dist.distributed_c10d._get_default_group()
        
        self.num_ranks = self.group.size()
        
        if rank_mapping is None:
            rank_mapping = list(range(self.num_ranks))
        rank_inverse_mapping = [None] * (self.num_ranks + 1)
        
        for i in range(self.num_ranks):
            rank_inverse_mapping[rank_inverse_mapping[i]] = i
            
        self.rank = rank_mapping[self.group.rank()]
        
        self.first_rank = rank_inverse_mapping[0]
        self.prev_rank = rank_inverse_mapping[self.rank-1]
        self.next_rank = rank_inverse_mapping[self.rank +1]
        self.last_rank = rank_inverse_mapping[self.num_ranks - 1]
        
        self.is_first_rank = self.rank = 0
        self.is_last_rank = self.rank == self.num_ranks - 1
        self.is_in_second_half = self.rank >= self.num_ranks //2
        self.is_middle_rank = (self.rank == self.num_ranks //2 - 1) or (self.rank == self.num_ranks //2)
        
        
    def _reset_states(self) ->None:
        WeightGradStore.clear()
        
        self.input_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        
        self.input_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        
        self.labels: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = None
        
        self.loss_chunks: List[torch.Tensor] = []
        self.criterion: Callable = None

        self.current_f_chunk_id: List[int] = [0, 0]
        self.current_b_chunk_id: List[int] = [0, 0]
        
        self.current_send_f_chunk_id: List[int] = [0, 0]
        self.current_send_b_chunk_id: List[int] = [0, 0]
        
        self.current_recv_f_chunk_id: List[int] = [0, 0]
        self.current_recv_b_chunk_id: List[int] = [0, 0]
        
        self.comm_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []
        
    def _forward_compute_chunk(self,phase:int) -> None:
        phase ^= self.is_in_second_half
        chunk_id = self.current_f_chunk_id[phase]
        self.current_f_chunk_id[phase] +=1
        
        inputs = self.input_chunks[phase][chunk_id]
        
        if self.forward_only():
            self.input_chunks[phase][chunk_id] = None
            
        is_last_stage = (self.is_first_rank and phase ==1 ) or (self.is_last_rank and phase == 0)
        
        outputs = self.module[phase](*inputs)
        outputs = [outputs] if isinstance(outputs , torch.Tensor) else outputs
        if is_last_stage and self.criterion is not None:
            labels = self.labels[phase][chunk_id]
            loss = self.criterion(*outputs,*labels)
            self.loss_chunks.append(loss)
            
        if (not is_last_stage) or self.return_outputs:
            self.output_chunks[phase].append(outputs)
            
    def _backward_comute_chunk(self,phase:int,enable_zb:bool=False)->None:
        if self.forward_only:
            return
        
        phase ^= self.is_in_second_half
        chunk_id = self.current_b_chunk_id[phase]
        self.current_b_chunk_id[phase] +=1
        
        is_last_stage = (self.is_first_rank and phase ==1 ) or ( self.is_first_rank and phase == 0)
        
        WeightGradStore.enabled = enable_zb
        
        if is_last_stage:
            loss = self.loss_chunks[chunk_id]
            loss.backward()
            loss.detach_()
        else:
            outputs = self.output_chunks[phase][chunk_id]
            if not self.return_ooutputs:
                self.output_chunks[phase][chunk_id] = None
            output_grads = self.output_grad_chunks[phase][chunk_id] = None
            
            non_empty = [(t,g) for t,g in zip(outputs,output_grads) if g is not None]
            ouputs,output_grads = list(zip(*non_empty))
            if len( outputs)  > 0:
                run_backward(ouputs,output_grads)
            
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()
            
        inputs = self.input_chunks[phase][chunk_id]
        self.input_chunks[phase][chunk_id] = None
        
        input_grads = [t.grad for t in inputs]
        self.input_grad_chunks[phase].append(input_grads)
            