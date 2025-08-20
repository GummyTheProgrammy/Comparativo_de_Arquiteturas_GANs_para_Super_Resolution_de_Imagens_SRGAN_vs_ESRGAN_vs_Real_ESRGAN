[Acesse o Artigo Completo Aqui](https://www.academia.edu/143524296/Compara%C3%A7%C3%A3o_de_Arquiteturas_GAN_para_Super_Resolu%C3%A7%C3%A3o_de_Imagens_SRGAN_ESRGAN_e_Real_ESRGAN)

# Comparativo de Arquiteturas GANs para Super-Resolution de Imagens: SRGAN vs. ESRGAN vs. Real-ESRGAN

[cite_start]Este repositório contém o notebook de Jupyter `Comparativo de Arquiteturas GANs para Super-Resolution de Imagens: SRGAN vs. ESRGAN vs. Real-ESRGAN.ipynb`, que explora e compara o desempenho de diferentes arquiteturas de Redes Adversariais Generativas (GANs) na tarefa de Super-Resolution de Imagens[cite: 1].

## Objetivo do Projeto

[cite_start]O objetivo principal deste projeto é **avaliar a capacidade de SRGAN, ESRGAN e Real-ESRGAN em recuperar detalhes finos e gerar texturas foto-realísticas** a partir de imagens de baixa resolução[cite: 1]. [cite_start]Este trabalho foi desenvolvido no contexto de um projeto universitário, com adaptações para viabilizar a execução no Google Colab[cite: 1]. [cite_start]Assim, foram utilizados um subconjunto do dataset, tamanhos de imagem reduzidos e um número menor de épocas de treinamento[cite: 1].

## Modelos e Arquiteturas Comparadas

[cite_start]O notebook detalha a implementação e comparação das seguintes arquiteturas GAN para Super-Resolution[cite: 1]:

* **SRGAN (Super-Resolution Generative Adversarial Network)**:
    * [cite_start]**Gerador (G_SRGAN)**: Utiliza uma rede residual profunda (ResNet) com skip-connections, blocos residuais contendo camadas convolucionais, Batch Normalization (BN) e ativação ParametricReLU (PReLU)[cite: 1]. [cite_start]Emprega convolução sub-pixel para upsampling[cite: 1].
    * [cite_start]**Discriminador (D_SRGAN)**: Composto por oito camadas convolucionais com filtros crescentes, ativação LeakyReLU (alpha=0.2) e convoluções strided para reduzir a resolução, finalizando com camadas densas e uma função Sigmoid para probabilidade de ser real[cite: 1].

* **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)**:
    * [cite_start]**Gerador (G_ESRGAN)**: Melhora o SRGAN substituindo os blocos residuais por Residual-in-Residual Dense Blocks (RRDB)[cite: 1]. [cite_start]Remove todas as camadas de Batch Normalization (BN) para evitar artefatos e melhorar a generalização[cite: 1]. [cite_start]Utiliza Residual Scaling e inicialização menor para treinar redes mais profundas[cite: 1].
    * [cite_start]**Discriminador (D_ESRGAN)**: Adota o **Relativistic Discriminator (RaGAN)**, que prevê se uma imagem é *relativamente* mais realista do que outra[cite: 1].

* **Real-ESRGAN (Real-World Blind Super-Resolution with Pure Synthetic Data)**:
    * [cite_start]**Gerador (G_RealESRGAN)**: Mantém a mesma arquitetura do gerador do ESRGAN (RRDB)[cite: 1]. [cite_start]Para fatores de escala menores, pode usar uma operação de pixel-unshuffle na entrada para otimizar memória e computação[cite: 1].
    * [cite_start]**Discriminador (D_RealESRGAN)**: Melhora o discriminador VGG-style do ESRGAN para um design **U-Net com skip connections**, fornecendo feedback per-pixel detalhado ao gerador[cite: 1]. [cite_start]Emprega **Spectral Normalization (SN)** para estabilizar o treinamento e aliviar artefatos de super-nitidez[cite: 1].

## Funções de Perda e Otimizadores

[cite_start]O projeto utiliza o otimizador **Adam** para o treinamento das GANs[cite: 1]. [cite_start]As funções de perda incluem[cite: 1]:

* **Perda de Conteúdo (Content Loss)**:
    * [cite_start]**MSE Loss** (Mean Squared Error): Usada em métodos PSNR-orientados, embora possa resultar em imagens suavizadas[cite: 1].
    * [cite_start]**L1 Loss** (Mean Absolute Error): Comum em ESRGAN e Real-ESRGAN[cite: 1].
    * [cite_start]**Perceptual Loss (VGG Loss)**: Baseada em camadas de ativação de uma rede VGG pré-treinada[cite: 1].
    * [cite_start]**Dual Perceptual Loss (DP Loss)**: Proposta para ESRGAN, combinando VGG Loss com uma **ResNet Loss**[cite: 1].

* **Perda Adversarial (Adversarial Loss)**:
    * [cite_start]**Binary Cross-Entropy (BCE)**: Para discriminadores padrão, diferenciando imagens reais das geradas[cite: 1].
    * [cite_start]**Relativistic Average GAN (RaGAN) Loss**: Usada em ESRGAN e Real-ESRGAN, com o gerador se beneficiando de gradientes tanto de dados gerados quanto reais[cite: 1].

* [cite_start]**Kullback-Leibler Divergence (KLD)**: Mencionada como uma perda relevante para Variational Autoencoders (VAEs), mas **não é uma perda primária** para as arquiteturas GAN puras focadas em super-resolution descritas aqui[cite: 1].

## Treinamento e Mecanismos Anti-Overfitting

[cite_start]O processo de treinamento envolve[cite: 1]:

* [cite_start]**Pré-treinamento do Gerador**: Com uma perda pixel-wise (ex: L1 Loss) para obter resultados iniciais mais agradáveis[cite: 1].
* [cite_start]**Treinamento Adversarial**: Alternando atualizações entre o Gerador e o Discriminador[cite: 1].

[cite_start]Mecanismos para prevenir o overfitting incluem[cite: 1]:

* [cite_start]**Batch Normalization**: ESRGAN e Real-ESRGAN removem BN no gerador para evitar artefatos e melhorar a generalização[cite: 1].
* [cite_start]**Regularização L2 (Weight Decay)**: Adicionada ao otimizador para penalizar pesos grandes[cite: 1].
* [cite_start]**Early Stopping**: Monitorar uma métrica de validação e parar o treinamento quando ela não melhorar por um certo número de épocas[cite: 1].

[cite_start]Para mitigar o **problema do gradiente desvanecente**, as soluções aplicadas nas GANs para SR incluem[cite: 1]:

* [cite_start]**Redes Residuais (ResNets) / Skip Connections**: Permitem que o gradiente flua diretamente através de atalhos[cite: 1].
* [cite_start]**Funções de Ativação (ReLU, LeakyReLU, PReLU)**: Evitam a saturação em valores positivos[cite: 1].

## Avaliação e Comparação

[cite_start]A avaliação dos modelos é realizada usando[cite: 1]:

* **Métricas de Distorção (Pixel-wise)**:
    * [cite_start]**PSNR (Peak Signal-to-Noise Ratio)**: Usado, mas não reflete bem a qualidade perceptiva humana[cite: 1].
    * [cite_start]**SSIM (Structural Similarity Index)**: Tenta medir a similaridade estrutural[cite: 1].

* **Métricas Perceptivas**:
    * [cite_start]**MOS (Mean Opinion Score)**: Testes subjetivos com avaliadores humanos[cite: 1].
    * [cite_start]**LPIPS (Learned Perceptual Image Patch Similarity)**: Correlaciona-se melhor com a percepção humana[cite: 1].
    * [cite_start]**Perceptual Index**: Combina Ma’s score e NIQE[cite: 1].

[cite_start]O notebook inclui **comparação quantitativa** em tabelas (PSNR, SSIM) e **comparação qualitativa** mostrando imagens lado a lado (LR de entrada, SR de cada modelo, e HR original)[cite: 1].

## Otimização e Hiperparâmetros

[cite_start]Conceitos como **Grid Search e Random Search** são discutidos como métodos para encontrar a melhor combinação de hiperparâmetros[cite: 1].

## Salvando Modelos

[cite_start]O projeto demonstra como salvar **checkpoints** do modelo e como salvar o **modelo final** com o melhor desempenho[cite: 1].

---

**DATASET DISPONÍVEL EM:**
[https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)