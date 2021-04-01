[toc]

# 华为NOVA模型解读

## 背景

> 序列推荐系统的目标是基于用户的历史行为对用户的兴趣进行建模，从而进行时间相关的个性化推荐。早期模型，比如CNN和RNN等深度学习方法在推荐任务中都取得了显著的提升。
>
> 近年来，BERT框架由于其在处理序列数据采用的self-attention机制，在许多任务上都取得了SOTA的效果，如Bert4Rec。但是BERT框架也存在局限性：只考虑自然语言符号的一个输入源，其最初设计为只接受一种类型的输入（即wordId），限制了side信息的使用。
>
> 在BERT框架下如何充分利用side信息仍然是一个悬而未决的问题。side信息，如商品的类别或tag标签，在进行更全面的描述和更好的推荐中具有重要意义。一些简单的方法直接将不同类型的side信息融合到商品embedding中，通常只带来很少甚至负面的影响。
>
> 因此，本文在BERT框架下提出了一种有效利用边信息的无创自注意机制（NOVA，NOninVasive self-Attention mechanism），充分利用side信息来产生更好的注意力分布，而不是直接改变商品embedding(这会导致信息泛滥)。
>
> 本文的核心贡献有三点：
>
> 1. 提出了NOVA-BERT框架，该框架可以有效地利用各种side信息进行序列化的推荐任务；
> 2. 提出了非侵入（non-invasive）的self-attention机制（NOVA），这是一种新的设计，可以实现对复合序列数据的self-attention；
> 3. 基于可视化给出了模型的可解释性。

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/1.png)

## 模型结构

### problem statement

> 给定一个用户的历史交互行为，序列推荐要求预测用户下一个交互的商品或行为，可以表示为：

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/2.jpg)

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/3.jpg)

### side information

> side信息是为推荐系统提供额外有效的信息，可以分为两种类型：商品相关信息或行为相关信息。
>
> - 基于商品的side信息是商品的固有信息，除了商品ID，还包括price、商品的日期、生产者等等
> - 基于行为相关的side信息与用户的交互有关，例如操作类型（**购买、打分**）、执行时间或用户反馈分数，每个交互的顺序（即原始BERT中的位置ID）也可以作为一种行为相关的side信息
>
> 加入side信息后，用户的历史交互行为可表示为：

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/5.jpg)

### BERT and Invasive Self-attention

> BERT4Rec是第一次将BERT框架用于序列化推荐任务的，并且取得了SOTA的效果。在BERT4Rec中，Item表示为向量，一些商品被随机mask掉，训练中采用multi-head self-attention机制recover这些商品向量：

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/5.jpg)

> 为了更好地利用side信息，传统方法经常会使用分开的embedding层来将side信息进行编码，然后将它们fuse到item ID的embedding中，融合的方法主要有：
>
> - Summation
> - Concatenation
> - Gated Sum
>
> 此类方法为Invasive的方法，因为它们改变了item原始的embedding表示。

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/6.jpg)

> 之后叠加self-attention机制不断更新表示层

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/7.jpg)

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/8.jpg)

> self-attention操作是位置不变的函数，所以此处将位置embedding编码加入其中，BERT4Rec仅仅是将位置信息作为了side信息，并且使用addition作为fusion函数。

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/9.jpg)

### Non-invasive Self-attention(NOVA)

> BERT框架是一个堆叠self-attention层的自动编码器，相同的embedding映射用于编码商品ID和解码还原的向量表示。因此，invasive方法存在复杂嵌入空间的缺点，因为item ID不可逆地与其他side信息融合，混合item id和side信息可能会使模型难以解码item ID。因此本文提出了一种新的方法，即non-invasive自注意（NOVA），在利用side信息对序列进行更有效建模的同时，保持嵌入空间的一致性。其思路是：
>
> - 修改自注意机制，仔细控制信息源，即Q、K和V

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/10.jpg)

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/11.jpg)

### Fusion Operations

> NOVA和invasive方法在使用side信息下的不同在于，NOVA将其作为一个辅助的并通过fusion函数将side信息作为Keys和Querys输入。本文中采用的fusion函数主要有以下三种：

- addition

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/12.jpg)

- concat

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/13.jpg)

- gating

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/14.jpg)

###  NOVA-BERT

> 每个NOVA层接受两个输入：side信息和item表示序列，然后输出相同的更新表示，再将这些表示输送送到下一层。对于第一层的输入，商品表示是纯item ID嵌入。由于只使用side信息作为辅助来学习更好的注意分布，side信息不会随着NOVA层传播，为每个NOVA层提供相同的side信息集。

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/15.jpg)

## 实验结果

### 效果比较

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/16.jpg)

> - NOVA-BERT的效果比其它的都要好；
> - 与Bert4Rec仅利用位置ID相比，invasive式方法使用了多种side信息，但改进非常有限甚至没有正向效果。相反，NOVA-BERT方法能有效地利用side信息，性能稳定，优于其他方法。
> - 在实验中发现越大的denser数据集，模型提升的幅度会下降；我们假设，在语料库更为丰富的情况下，这些模型甚至可以从商品的上下文中学习到足够好的商品embedding，从而为辅助信息的补充留下更小的空间。
> - NOVA-BERT的鲁棒性是非常好的；不管使用什么fusion函数，Nova-Bert的效果一直比baseline模型要好；
> - 最佳融合函数可能取决于**数据集**。一般来说，gating方法具有很强的性能，这可能得益于其可训练的gating机制。结果还表明，对于实际部署，融合函数的类型可以是一个超参数，并应根据数据集的内在属性进行调整，以达到最佳的在线性能。

### 不同side信息的贡献

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/17.jpg)

> - None：是原始的Bert+position ID；
> - 商品相关和行为相关的side信息并未带来准确率的明显提升；
> - 如果结合了与行为相关的side信息，则改进的效果明显大于其中任何一个带来的改进的总和；也就是说不同类型的side信息并不是独立的；
> - NOVA-BERT从综合信息中获益更多，能够充分利用丰富的数据，并且不受信息的困扰；

### Attention可视化分布

![image](https://github.com/ShaoQiBNU/NOVA/blob/main/img/18.jpg)

> - NOVA-BERT的注意力得分在局部性方面表现出更强的模式，大致集中在对角线上。另一方面，在基线模型的图中没有观察到。对整个数据集的观察，这种对比是普遍存在的。side信息导致模型在早期层次形成更明显的attention。
> - 实验结果表明，NOVA-BERT算法以side信息作为计算attention矩阵的辅助工具，可以学习目标的注意分布，从而提高计算的准确性。

