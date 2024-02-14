

## 1. Basic model parameters
A model usually consist of (embedding, encoder/decoder, lm_head).
### embedding 
embedding layer is a look-up table, with parameters:

    N = n_vocab * d_model
    (n_vocab is the size of vocab dict, d_model is the embedding size)

In some cases, embedding layer got frozen and un-trainable.

### encoder/decoder
with each multi-head attention block, it can be represented as:

    N(Q,K,V) = 3*(d_model*d_head*n_head) + d_head*n_head*d_model 
    (d_model is the block input embedding size, d_head is dimension of QKV, n_head is number of heads)

after multi_head attention block, followed by 2-linear layers mlp:

    N(mlp) = d_model * d_ffn + d_ffn * d_model
    (d_model is mlp input embedding size, d_ffn is the projected ffn embedding size)

### lm_head
simple linear layer to project feature space into probability space:

    N(lm_head) = d_model * n_vocab
    (d_model is the lm_head input embedding size, n_vocab is vocab dict size)

### model
Thus, the total model parameters size becomes:

    N(model) = n_vocab * d_model + n_blocks*(4(d_model*d_head*n_head) + 2(d_model*d_ffn)) + d_model * n_vocab

Use llama2-7B as example, where n_vocab=32000, d_model=4096, d_model=d_head*n_head,
d_ffn=11008, n_blocks=32. 

    N(llama2-7b) = 6738149376, almost 7b parameters.


## 2. Basic model memory usage
memory usage during model training can be divided into 2 parts: fixed model parameters occupancy and dynamic
intermediate activations occupancy. 
### parameters occupancy
In training, due to Adam widely used, the memory usage is actually far more than model parameters size.

    1) use fp32 datatype
    O(parameters) = (N_parameters + N_grad + N_adamm + N_adamv) * 4 [bytes]
    (N_paramters/N_grad/N_adamm/N_adamv=N, N is the number of model parameters)

    total memory usage is 16N (bytes). With a llama2-7b, it becomes 104G

    2) use mixed datatype
    O(parameters) = N_parameters*2 + N_grad*2 + N_adamm*4 + N_adamv*4 + N_parameters*4 [bytes]
    (N_paramters/N_grad/N_adamm/N_adamv=N, N is the number of model parameters)

    total memory usage is 16N (bytes). With a llama2-7b, it becomes 104G

### activations occupancy
PyTorch describe deep learning model as computation graph. Operation receive input tensor, do computation, 
output activation tensor to downstream operations. Those activation tensor will be remained, and occupy memory. 
(Note inside an operation, some temp activations get created but released quickly, thus becomes negligible.)
(Note we assume mixed precision mode below)

#### Forward

    1.embedding layer is one lookup operation
        O(embedding) = (n_batch * n_seq * d_model) * 2 [bytes]
        (n_batch is batch size, n_seq is input token length, d_model is embedding size)

    2.encoder/decoder
    with each multi-head attention block, the operations include:
    2.1 QKV projection
        O(qkv_proj) = 3*(n_batch * n_seq * n_head * d_head) * 2 [bytes]
    2.2 QK cross attention
        O(qk_atten) = (n_batch * n_head * n_seq * n_seq) * 2 [bytes]
    2.3 Query
        O(v) = (n_batch * n_seq * n_head * d_head) * 2 [bytes]
    2.4 Linear project 
        O(l) = (n_batch * n_seq * d_model) * 2 [bytes]
    2.5 Mlp layer
        O(mlp) = n_batch * n_seq * d_ffn + n_batch * n_seq * d_ffn + n_batch *  n_seq * d_model 
        (refers to first linear, relu, and second linear)         
    2.5 Norm
        O(norm) = n_norms * (n_batch * n_seq * d_model) * 2 [bytes]

    (n_batch is batch size, n_seq is input token length, d_head is dimension of QKV, 
     n_head is number of heads, d_model is embedding size, n_norms is number of norm layers)

    3. lm_head
        O(lm_head) = (n_batch * n_seq * n_vocab) * 2 [bytes]

    O(forward) = (n_batch * n_seq * d_model +\
                  n_layers*(n_batch * n_seq * (4 * n_head * d_head + n_head * n_seq + 2*d_model + 2*d_ffn) +\
                    n_norms * (n_batch * n_seq * d_model)) +\
                  (n_batch * n_seq * 2*n_vocab)) * 2 [bytes]

#### Backward   

    O(backward) = O(forward)

Use llama2-7b as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
n_blocks=32, n_batch=12, n_seq=64
    
    O(llama2) = O(parameters) + O(forward) + O(backword) = 104G + 3G + 3G
    (However, n_seq and n_batch now get larger and larger! for 128 batch and 512 sequence length, 
     O(forward) surges to 240G and O(llama2) surges to 580G!)



## 3. Lora model memory usage          
With Lora, the trainable parameters significantly reduced, usually <0.1% of model parameters.
### parameters occupancy

    since few trainable parameters, the gradients/adam states parameters are negligible.

    O(parameters) = N_parameters*2 + N_grad*2 + N_adamm*4 + N_adamv*4 + N_lora*4 [bytes]
    (N_paramters=N, N_grad/N_adamm/N_adamv/N_lora=N*0.1%, N is the number of model parameters)

    total memory usage is ~2N (bytes). With a llama2-7b, it becomes 14G


### activations occupancy
#### Forward

    O(forward) = (n_batch * n_seq * d_model +\
                  n_layers*(n_batch * n_seq * (4 * n_head * d_head + n_head * n_seq + 2*d_model + 2*d_ffn) +\
                    n_norms * (n_batch * n_seq * d_model)) +\
                  (n_batch * n_seq * 2*n_vocab)) * 2 [bytes]

#### Backward

    O(forward) becomes negligible as few trainable parameters.

Use llama2-7b-lora as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
n_blocks=32, n_batch=12, n_seq=64
    
    O(llama2_lora) = O(parameters) + O(forward) + O(backword) ~= 14G + 3G, about 17G.


## 4. Deepspeed+lora model memory usage
Deepspeed is a deep learning optimization library for distributed training. For straight forward quantitative
analyze on memory usage, we utilze zero3 + lora as example. 
### one gpu
As known, deepspeed is for distributed training. However, it would also optimize the single GPU training pipeline.
1) The deepspeed init would optimize the model, mainly cast fp32 to fp16 to reduce parameters size.
2) deepspeed would also integrate memory fragmentations, boost memory efficiency.

Use llama2-7b-lora as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
n_blocks=32, n_batch=12, n_seq=64. As lora leaves few optimizer states, ignore.
   
    Raw model: 26G (fp32)
    Training MA peak: 38G

    After deepspeed init: 13G (fp16)
    Training MA peak: 21G

### multi gpu (one node)
zero3 would partition model weights as well as optimizer states onto gpus, ensuring each gpu keeps only a section.
1) partition linearly reduce memory usage for the model parameters and optimizer state, because ZeRO-3 partitions 
those across the available GPUs.
2) usage at other points represents consumption by input data, activations, gradients, intermediate buffers, and 
fragmentation resulting from PyTorch computation. ZeRO3 does not affect the memory consumed by these buffers.
3) collect partitioned model parameters and optimizer states take extra memory cost.

Use llama2-7b-lora as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
n_blocks=32, n_batch=12, n_seq=64. As lora leaves few optimizer states, ignore. On 4 GPUs.

    Raw model: 26G (fp32)
    After deepspeed init: 6G per gpu (fp16)
    Training MA peak: 26G
    (Note: above example shows 20G intermediate memory usage. compared to zero2, conclude that extra 10G is used for
     collect process)







