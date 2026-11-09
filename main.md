
The debate between latent and pixel space in image generation has argued too long. The triditional latant diffusion model enpress picel into a more intight space to reduce the compute comsume and let diffusion generator learn easier. The pixel 拥簇者 believe that deal with pixel derictly can archive a higher performence without any loss of internal representation.
Now we present The X-drifting model, which drifting from latent to picel, to end the 相反的 research paths. X-drifting have the both adavantage of latent and pixel, who still evolatue from a 精巧的 represention but is supervisd by pixel face to face.(实验结果和website) 


### Introduction

Image generation has take a long way. VAEs[][](绑定了重建与生成) .... GANs[][] bring a more insight view from game theory. And then sorce-based model[][]include diffusion and flow-matching 横空出世， becoming another name of generitive model almost by its impressive visial preformence. {A novel pradise named drifting model[][] challanged with the past resently. Its evolation during training propose a new native one-step generation path.}
（感觉没有一开始就指出矛盾）

（引入问题）Among the prevailing latent diffusion model, VAE serves as a compresser basicly, but the symmtirc design of encoder and decoder introduce a unequity of generating  the midddle latents and reconstrating the final iamges.A path is to produce more powerful autoencoders and deocders.{wan-vae, hunyuan-vae} Now a new fasion is mask the frozen pretrain unsupervised foundation models as the encoder, as they saying the latents are more sementic, the generation is faster and better.


In the same time, some works are vigorously promote pixel diffusion model.Thay adhered to the will of "Back to Basic" and generated image without any hidden representation.It is a nature way to broke the unequty becuase it eliminated the constration competetliy.Though these works has a huge improvement in computation and generation quity, but now pixel diffusion model is still suffer from （训练周期长不收敛 and 色彩偏移，模糊）.

（提出问题与对比工作）
So a simple and marvelous idea is can we 一方面享受latent的**生成加速与压缩便利**，（need proof）一方面直接监督pixel避免重建损失且享受pixel的优化红利。the prior classic works can not do becuse the noise space and the output space mast be the same, because the learning trget is the interpolation of them. CrossFlow is the frist one to deal with this problem by removing the velocity term using a fantasitc math tool. But the main proposed methed is actualy consistent model type, it enfore model predivct the final result at any time.Snd the final result is inpreved by gan and precuptual loss mostly. 

（我们的探索）
We present the Drifting-X model, a cross-space generative model.We find and proof the drifting model is native suit for the unmatching of input and output space. So we introduce it into the same vanilla vision transformer to compere CrossFlow fairly. We proof the drifting methed is more outstanding though a 2d toy demo. Also we add different autoencoder to proof our path can leavrage the latent's 优化 fristly.We achive ----

（总结）

contribution
1. 将drfiting model推广到Cross space场景
2. 我们propose一条新的生成范式：负责压缩与提供更好的学习表征的Encoder依然很重要，但不在需要负责重建的decoder托生成的后腿(然后测量RAE、PAE刷点)
3. 我们通过实验证明drifting在CrossSpace场景下最优，超过了集成others的CrossFlow尽管主要依赖GAN实现CrossSpace,成为领域sota。


---
消融实验提示：对于pixel空间带来的利益，我们可以使用f-loss和dino作为drifting场的提取器等体现
---