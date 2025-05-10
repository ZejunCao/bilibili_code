bilibili视频链接：https://www.bilibili.com/video/BV1jcVHzuE8v

知乎博客链接：https://zhuanlan.zhihu.com/p/1901704483446187870

CSDN博客链接：https://blog.csdn.net/qq_41496421/article/details/147682009

# 引言

注意力机制作为大语言模型的核心组件，这么多年从最开始的 MHA 到现在最常用的 MQA、GQA，最主要的目的都是为了节省kv cache的大小。

MHA每一层需要存储【序列长度注意力头数每头维度】的大小，而MQA让每个头的k共享，需要存储的维度直接降低为【序列长度1每头维度】，但后面发现这样降的太多就导致性能下降，所以设计出了一种折中方案。GQA自定义多少个头共享一个k，最终维度变为【序列长度组数每头维度】

![MHA、MQA、GQA对比图](https://s3.bmp.ovh/imgs/2025/05/02/af5fe3eb5d013b2d.png)

以下给出了GQA的计算结构图，这里设置的组数为4，MHA和MQA就是将这个组数修改为注意力头数或1。

![GQA数据流向图](https://s3.bmp.ovh/imgs/2025/05/02/6db3ca6846ac2e30.png)

MLA 借鉴了LoRA的思路，使用一个降维矩阵将隐层维度降低，然后存储为kv cache，在注意力计算时，使用一个升维矩阵将kv cache升维，从而达到节省kv cache的目的，而且由于升降维矩阵的存在，性能并不会降低（实验证明反而会提高）。

到这个时候，网上已经有很多关于MLA理论的讲解了，但 MLA 听着简单，就是注意力降维、解耦旋转矩阵、吸收矩阵，但你真得搞懂它的内部细节了吗。
MLA 内部涉及10多个矩阵，绕来绕去都晕了，每一步具体怎么切分的，怎么转化维度的，如果让你清晰的描述出来，可能也会很难吧。

本文结合网络图和代码，一步一步详细讲解MLA都做了什么，那么多矩阵都是做什么用的，还请耐心观看。

针对每一个token的注意力计算，都是一个重复的过程，那我们就取中间的一步进行模拟MLA计算。注意这里的维度大小我直接按照deepseek的参数写具体值，这样更为清晰。本文中的矩阵及向量命名都遵守deepseek的命名。

# MLA 数据流向

## MLA朴素版

<font color="grayblue">首先介绍MLA的常规计算</font>

![MLA 朴素版数据流向图](https://s3.bmp.ovh/imgs/2025/05/11/e43c9410aa3800b9.png)

1. 输入：首先注意力计算 forward 函数会输入隐层向量 hidden_state，记作$h_t$，它的维度是[1, 7168]，因为推理时是一个token一个token进行处理的。
   
   还会输入 kv cache，记作$c^{KV}$，它的维度是[n-1, 512]，n-1是历史序列长度。
   
   由于旋转位置编码解耦，所以还要输入一个 $k^R$，它的维度是[n-1, 64]，**这里k的旋转位置编码在各个头是共享的，所以不需要128*64个**。
2. 计算q：首先基于 $h_t$ 计算当前 token 的 $q_t^C$ 和 $q_t^R$，即拆分成没有rope和带rope的。先将 $h_t$ 进行降维，得到 $c_t^Q=h_tW^{DQ}$，它的维度是 [1, 1536]。
   
   然后与 $W^{UQ}$ 相乘，得到 $q_t^C=c_t^QW^{UQ}$，它的维度是 [1, 128*128]，代表128个头，每个头128个维度。
   
   同理，$q_t^R=c_t^QW^{QR}$ 的维度是 [1, 128*64]。
3. 计算c：然后将当前 token 转化成 c 作为 kv cache。
   
   直接将 $h_t$ 降维，得到 $c_t^{KV}=h_tW^{DKV}$，它的维度是 [1, 512]。将其与历史 kv cache 拼接，记作$c^{KV}$，它的维度是 [n, 512]。同时将本次的  $c^{KV}$ 存储下来，用于下次计算。
4. 计算kv：处理 kv cache 即 $c^{KV}$，得到可计算的 k 和 v。
   
   $k^C=c^{KV}W^{UK}$，它的维度是 [n, 128*128]。
   
   $v^C=c^{KV}W^{UV}$，它的维度是 [n, 128*128]。

   $k_t^R=h_tW^{KR}$，它的维度是 [1, 64]，与输入的 $k_{pe} cache$ 拼接到一起，得到 $k_t^R$，维度是 [n, 64]，注意这里每个头之间是共享的，所以不需要128*64个。
但是在后续注意力计算的时候需要维度广播，复制出128份。
5. 计算注意力权重：
   
   $attn^C = q_t^C(k^C)^T$，它的维度是 [n, 128]。
   
   $attn^R = q_t^R(k^R)^T$，它的维度是 [n, 128]。
   
   $attn = attn^C + attn^R$
   
   $attn\_weight = softmax(\frac{attn}{\sqrt{d}})$
   
   这里带rope和不带rope的注意力是分开算的，根据矩阵的性质，分开计算再相加与合并后计算的结果是相同的。
   
   其等价于：
   
   $attn = [q_t^C; q_t^R]([k^C; k^R])^T$
6. 与v相乘：$attn\_output = attn\_weight * v^C$
7. 最终输出：$output = attn\_output * W^{O}$

总体公式为（当前token转化成c需要单独计算，且忽略rope的部分）：

$$
output = softmax(\frac{(h_tW^{DQ}W^{UQ})(c^{KV}W^{UK})^T}{\sqrt{d}}) (c^{KV}W^{UV}) W^{O} \\
= softmax(\frac{h_tW^{DQ}W^{UQ}W^{UK^T}c^{KV^T}}{\sqrt{d}})c^{KV}W^{UV}W^{O}
$$

## MLA 吸收矩阵版

<font color="grayblue">接下来介绍MLA吸收矩阵的计算方式</font>

![MLA 吸收矩阵数据流向图](https://s3.bmp.ovh/imgs/2025/05/11/c68ab8efe28309a0.png)

上面总体公式中 $W^{UQ}W^{UK^T}$ 是挨着的，$W^{UV}W^{O}$ 也是挨着的，所以可以提前合并成一个矩阵，记作$W^{UQK}$和$W^{UVO}$，这样每次推理就不用进行两次矩阵运算了，加快推理速度，这个就叫做吸收矩阵 **(absorb matrix)**。

那吸收之后的总体公式变为：

$$
output = softmax(\frac{h_tW^{DQ}W^{UQK}c^{KV^T}}{\sqrt{d}})c^{KV}W^{UVO}
$$

那整体计算流程就变成了：

1. 输入：与常规相同
2. 计算q：还是首先基于 $h_t$ 计算当前 token 的 $q_t^C$ 和 $q_t^R$。首先还是将$h_t$降维，得到$c_t^Q=h_tW^{DQ}$，它的维度是 [1, 1536]。
   
   $q_t^R$与常规相同：$q_t^R=c_t^QW^{QR}$ 的维度是 [1, 128*64]。
   
   $q_t^C$直接一步到位乘以吸收矩阵：$q_t^C=c_t^QW^{UQK}=h_tW^{DQ}W^{UQK}$，它的维度是 [1, 128*512]。
3. 计算c：与常规相同
4. 计算kv：这步删除掉处理 $k^C$ 和 $v^C$ 的步骤，不需要提前分解$c^{KV}$了，但 $k^R$ 与常规相同。
5. 计算注意力权重：
   
   $attn^C = q_t^C(c^{KV})^T$，它的维度是 [n, 128]。
   
   $attn^R = q_t^R(k^R)^T$，与常规一样，它的维度是 [n, 128]。
   
   $attn = attn^C + attn^R$
   
   $attn\_weight = softmax(\frac{attn}{\sqrt{d}})$
6. 与v相乘+最终输出：两步合为一步，$output = attn\_weight * c^{KV}W^{UVO}$

至此，MLA就介绍完了，现在你还能复述一遍 MLA 的计算流程吗？那些矩阵还能分得清吗？如果都能搞懂，说明你真得掌握了 MLA，可以去看看 flash MLA 了[狗头]。

# 参考资料

1. [https://arxiv.org/pdf/2405.04434](!https://arxiv.org/pdf/2405.04434)
2. [https://kexue.fm/archives/10091](!https://kexue.fm/archives/10091)
3. [https://mp.weixin.qq.com/s/E7NwwMYw14FRT6OKzuVXFA](!https://mp.weixin.qq.com/s/E7NwwMYw14FRT6OKzuVXFA)
4. [https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py#L682](!https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py#L682)
5. [https://github.com/flashinfer-ai/flashinfer/blob/738460ff82e2230ebcc8dff50e49e1d6278e011a/tests/test_mla_decode_kernel.py](!https://github.com/flashinfer-ai/flashinfer/blob/738460ff82e2230ebcc8dff50e49e1d6278e011a/tests/test_mla_decode_kernel.py)
