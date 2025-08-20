```markdown
# Comparativo de Arquiteturas GANs para Super-Resolution de Imagens: SRGAN vs. ESRGAN vs. Real-ESRGAN

Este repositório contém o notebook de Jupyter `Comparativo de Arquiteturas GANs para Super-Resolution de Imagens: SRGAN vs. ESRGAN vs. Real-ESRGAN.ipynb`, que explora e compara o desempenho de diferentes arquiteturas de Redes Adversariais Generativas (GANs) na tarefa de Super-Resolution de Imagens.

## Objetivo do Projeto

O objetivo principal deste projeto é **avaliar a capacidade de SRGAN, ESRGAN e Real-ESRGAN em recuperar detalhes finos e gerar texturas foto-realísticas** a partir de imagens de baixa resolução. Este trabalho foi desenvolvido no contexto de um projeto universitário, com adaptações para viabilizar a execução no Google Colab, que possui limitações de GPU, RAM e tempo de execução. Assim, foram utilizados um subconjunto do dataset, tamanhos de imagem reduzidos e um número menor de épocas de treinamento.

## Modelos e Arquiteturas Comparadas

O notebook detalha a implementação e comparação das seguintes arquiteturas GAN para Super-Resolution:

*   **SRGAN (Super-Resolution Generative Adversarial Network)**:
    *   **Gerador (G_SRGAN)**: Utiliza uma rede residual profunda (ResNet) com skip-connections, blocos residuais contendo camadas convolucionais, Batch Normalization (BN) e ativação ParametricReLU (PReLU). Emprega convolução sub-pixel para upsampling.
    *   **Discriminador (D_SRGAN)**: Composto por oito camadas convolucionais com filtros crescentes, ativação LeakyReLU (alpha=0.2) e convoluções strided para reduzir a resolução, finalizando com camadas densas e uma função Sigmoid para probabilidade de ser real.

*   **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)**:
    *   **Gerador (G_ESRGAN)**: Melhora o SRGAN substituindo os blocos residuais por Residual-in-Residual Dense Blocks (RRDB) e **removendo todas as camadas de Batch Normalization (BN)** para evitar artefatos e melhorar a generalização. Utiliza Residual Scaling e inicialização menor para treinar redes mais profundas.
    *   **Discriminador (D_ESRGAN)**: Adota o **Relativistic Discriminator (RaGAN)**, que prevê se uma imagem é *relativamente* mais realista do que outra, ajudando o gerador a criar texturas mais realistas.

*   **Real-ESRGAN (Real-World Blind Super-Resolution with Pure Synthetic Data)**:
    *   **Gerador (G_RealESRGAN)**: Mantém a mesma arquitetura do gerador do ESRGAN (RRDB). Para fatores de escala menores (ex: 2x, 1x), pode usar uma operação de pixel-unshuffle na entrada para otimizar memória e computação.
    *   **Discriminador (D_RealESRGAN)**: Melhora o discriminador VGG-style do ESRGAN para um design **U-Net com skip connections**, fornecendo feedback per-pixel detalhado ao gerador. Emprega **Spectral Normalization (SN)** para estabilizar o treinamento e aliviar artefatos de super-nitidez.

## Funções de Perda e Otimizadores

O projeto utiliza o otimizador **Adam** para o treinamento das GANs. As funções de perda incluem:

*   **Perda de Conteúdo (Content Loss)**:
    *   **MSE Loss** (Mean Squared Error): Usada em métodos PSNR-orientados, embora possa resultar em imagens suavizadas.
    *   **L1 Loss** (Mean Absolute Error): Comum em ESRGAN e Real-ESRGAN.
    *   **Perceptual Loss (VGG Loss)**: Baseada em camadas de ativação de uma rede VGG pré-treinada. ESRGAN usa features *antes* da ativação para resultados mais nítidos.
    *   **Dual Perceptual Loss (DP Loss)**: Proposta para ESRGAN, combinando VGG Loss com uma **ResNet Loss** (baseada em features de uma ResNet pré-treinada).

*   **Perda Adversarial (Adversarial Loss)**:
    *   **Binary Cross-Entropy (BCE)**: Para discriminadores padrão, diferenciando imagens reais das geradas.
    *   **Relativistic Average GAN (RaGAN) Loss**: Usada em ESRGAN e Real-ESRGAN, com o gerador se beneficiando de gradientes tanto de dados gerados quanto reais para aprender bordas e texturas mais detalhadas.

*   **Kullback-Leibler Divergence (KLD)**: Mencionada como uma perda relevante para Variational Autoencoders (VAEs), mas **não é uma perda primária** para as arquiteturas GAN puras focadas em super-resolution descritas aqui.

## Treinamento e Mecanismos Anti-Overfitting

O processo de treinamento envolve:

*   **Pré-treinamento do Gerador**: Com uma perda pixel-wise (ex: L1 Loss) para obter resultados iniciais mais agradáveis e evitar ótimos locais indesejados no treinamento adversarial.
*   **Treinamento Adversarial**: Alternando atualizações entre o Gerador e o Discriminador, onde o Gerador tenta "enganar" o Discriminador, e este tenta diferenciar imagens reais das geradas.

Mecanismos para prevenir o overfitting incluem:

*   **Batch Normalization**: Embora geralmente ajude, ESRGAN e Real-ESRGAN removem BN no gerador para evitar artefatos e melhorar a generalização.
*   **Regularização L2 (Weight Decay)**: Adicionada ao otimizador para penalizar pesos grandes.
*   **Early Stopping**: Monitorar uma métrica de validação (ex: LPIPS) e parar o treinamento quando ela não melhorar por um certo número de épocas.
*   `model.train()` e `model.eval()`: Métodos PyTorch para alternar o comportamento de camadas de regularização.

Para mitigar o **problema do gradiente desvanecente**, as soluções aplicadas nas GANs para SR incluem:

*   **Redes Residuais (ResNets) / Skip Connections**: Permitem que o gradiente flua diretamente através de atalhos, facilitando o treinamento de redes profundas.
*   **Funções de Ativação (ReLU, LeakyReLU, PReLU)**: Evitam a saturação em valores positivos.

## Avaliação e Comparação

A avaliação dos modelos é realizada usando:

*   **Métricas de Distorção (Pixel-wise)**:
    *   **PSNR (Peak Signal-to-Noise Ratio)**: Amplamente utilizado, mas não reflete bem a qualidade perceptiva humana.
    *   **SSIM (Structural Similarity Index)**: Tenta medir a similaridade estrutural.

*   **Métricas Perceptivas** (mencionadas, mas LPIPS foi omitido na execução para otimização de tempo):
    *   **MOS (Mean Opinion Score)**: Testes subjetivos com avaliadores humanos.
    *   **LPIPS (Learned Perceptual Image Patch Similarity)**: Correlaciona-se melhor com a percepção humana.
    *   **Perceptual Index**: Combina Ma’s score e NIQE.

O notebook inclui **comparação quantitativa** em tabelas (PSNR, SSIM) e **comparação qualitativa** mostrando imagens lado a lado (LR de entrada, SR de cada modelo, e HR original).

## Otimização e Hiperparâmetros

Conceitos como **Grid Search e Random Search** são discutidos como métodos para encontrar a melhor combinação de hiperparâmetros, embora a otimização em GANs possa ser complexa devido à instabilidade do treinamento.

## Salvando Modelos

O projeto demonstra como salvar **checkpoints** do modelo (estado dos pesos, otimizador, etc.) em intervalos regulares para recuperação do progresso, e como salvar o **modelo final** com o melhor desempenho.

---

**DATASET DISPONÍVEL EM:**
[https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
```