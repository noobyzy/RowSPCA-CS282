## yuzy `2021年4月30日 12:27`
Dear Dr. Wang:

My name is Ziyi Yu, an undergraduate student in School of Science and Technology(SIST), ShanghaiTech University, Shanghai, China. I am majoring in Computer Science.

I and my teammates would like to reproduce your paper "Upper bounds for Model-Free Row-Sparse Principal Component Analysis". We have encounted a few problems, and we would appreciate it if you would like to assist us in the following items.

1. Can you provide us with all the datasets you use? Especially the "Reddit" dataset.

2. If possible, can you provide us with the code?

3. The proof of Theorem 2, which claims that Algorithm 1: Local Search Method is a monotone decreasing algo, is really vague even in the appendix. Our concern is that the [V]_S in the objective function changes each iteration. Is there a more detailed version of the proof?

Thanks for your kindness and patience.

Sincerely

Ziyi Yu

## wang `2021年4月30日 23:05`
Dear Ziyi,

Thank you for interested in our paper. First, I would like to say we have an updated version of the ICML paper, please check this link: https://arxiv.org/pdf/2010.11152.pdf . 
You can find out the dataset at https://www2.isye.gatech.edu/~sdey30/publications.html. But we also attached the "Reddit" dataset to this email. 
We do not have code available online. The implementation is straightforward. To obtain dual (upper) bounds, we code constraints proposed in the paper and run it using Gurobi in Python. 
We have the new proof in Section 4 in the updated version.
Hope these work for you. 

Best regards,
Guanyi