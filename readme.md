Alguns programas que quero testar para compressão de imagens

## COMPRESSÃO 

# Comprimir todas as imagens em files/hific/hific-mi/original/ (GPU)

export CUDA_ROOT=$CONDA_PREFIX && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && export CUDA_HOME=$CONDA_PREFIX && export TF_FORCE_GPU_ALLOW_GROWTH=true && cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models && python tfci.py compress hific-mi none --base_folder files/hific

ou

./compress_gpu.sh hific-mi files/hific none

# Comprimir todas as imagens em files/hific/hific-mi/original/ (CPU)
python tfci.py compress hific-mi none --base_folder files/hific

# Comprimir arquivo específico (CPU)
python tfci.py compress hific-mi imagem.png --base_folder files/hific



## DECOMPRESSÃO 

./decompress_gpu.sh files/hific hific-mi none

# Descomprimir especificando modelo
python tfci.py decompress none --base_folder files/hific --model_folder hific-mi

# Descompressão com auto-detecção de modelo
python tfci.py decompress none --base_folder files/hific

# Descomprimir arquivo específico
python tfci.py decompress arquivo.tfci --base_folder files/hific --model_folder hific-mi



## ANÁLISE COMPRESSÃO



# LPIPS 360 com peso coseno
python compression_analysis.py --base_dir files --methods hific --metrics psnr lpips lpips360 --force_cpu --lpips360_weight_type cosine --lpips360_pole_weight 0.3

# Comparação LPIPS vs LPIPS 360 peso linear
python compression_analysis.py --base_dir files --methods hific --metrics psnr lpips lpips360 --force_cpu --lpips360_weight_type linear --lpips360_pole_weight 0.5

# LPIPS 360 com peso quadrático
python compression_analysis.py --base_dir files --methods hific --metrics lpips360 --force_cpu --lpips360_weight_type quadratic --lpips360_pole_weight 0.8


primeiro passo:

python -m hific.train --config mselpips_lo --ckpt_dir ckpts/mse_lpips_lo_200k --num_steps 200k --local_image_dir ../SUN360/train
            
segundo passo:            

python -m hific.train --config hific_lo --ckpt_dir ckpts/hific_mse_lpips_lo_200k --init_autoencoder_from_ckpt_dir ckpts/mse_lpips_lo_200k --num_steps 200k --local_image_dir ../SUN360/train
            

avaliar o modelo:

python -m hific.evaluate   --config hific   --ckpt_dir ckpts/hific_test   --out_dir evaluation_results/   --local_image_dir ../SUN360/test-10

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi --ckpt_dir ckpts/mse_lpips_lo_200k ckpts/mse_lpips_mi_200k ckpts/mse_lpips_hi_200k ckpts/hific_mse_lpips_lo_200k ckpts/hific_mse_lpips_mi_200k ckpts/hific_mse_lpips_hi_200k ckpts/mse_ssim_lo_200k ckpts/mse_ssim_mi_200k ckpts/mse_ssim_hi_200k ckpts/hific_mse_ssim_lo_200k ckpts/hific_mse_ssim_mi_200k ckpts/hific_mse_ssim_hi_200k ckpts/WSmse_WSssim_lo_200k ckpts/WSmse_WSssim_mi_200k ckpts/WSmse_WSssim_hi_200k ckpts/hific_WSmse_WSssim_lo_200k ckpts/hific_WSmse_WSssim_mi_200k ckpts/hific_WSmse_WSssim_hi_200k ckpts/gauss_WSmse_WSssim_lo_200k ckpts/gauss_WSmse_WSssim_mi_200k ckpts/gauss_WSmse_WSssim_hi_200k ckpts/gauss_hific_WSmse_WSssim_lo_200k ckpts/gauss_hific_WSmse_WSssim_mi_200k ckpts/gauss_hific_WSmse_WSssim_hi_200k --out_dir results/mselpips_lo results/mselpips_mi results/mselpips_hi results/hificlpips_lo results/hificlpips_mi results/hificlpips_hi results/msessim_lo results/msessim_mi results/msessim_hi results/hificssim_lo results/hificssim_mi results/hificssim_hi results/WSmsessim_lo results/WSmsessim_mi results/WSmsessim_hi results/WShificssim_lo results/WShificssim_mi results/WShificssim_hi results/gaussssim_lo results/gaussssim_mi results/gaussssim_hi results/gausshific_lo results/gausshific_mi results/gausshific_hi --group LPIPS LPIPS LPIPS HiFiCLPIPS HiFiCLPIPS HiFiCLPIPS SSIM SSIM SSIM HiFiCSSIM HiFiCSSIM HiFiCSSIM WSSSIM WSSSIM WSSSIM WSHiFiCSSIM WSHiFiCSSIM WSHiFiCSSIM GAUSS GAUSS GAUSS GAUSSHIFIC GAUSSHIFIC GAUSSHIFIC --local_image_dir ../CTC-360-resized --results_csv results/resultados_finais.csv

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi --ckpt_dir ckpts/SWHDC_WSmse_WSssim_lo_200k ckpts/SWHDC_WSmse_WSssim_mi_200k ckpts/SWHDC_WSmse_WSssim_hi_200k --out_dir results/SWHDC_lo results/SWHDC_mi results/SWHDC_hi --group SWHDC SWHDC SWHDC --local_image_dir ../CTC-360-resized --results_csv results/resultados_finais_new.csv

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi --ckpt_dir ckpts/SWHDC_WSmse_WSssim_256x512_lo_200k ckpts/SWHDC_WSmse_WSssim_256x512_mi_200k ckpts/SWHDC_WSmse_WSssim_256x512_hi_200k --out_dir results/SWHDC_256x512_lo results/SWHDC_256x512_mi results/SWHDC_256x512_hi --group SWHDC256x512 SWHDC256x512 SWHDC256x512 --local_image_dir ../CTC-360-resized --results_csv results/resultados_SWHDC_256x512.csv

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi --ckpt_dir ckpts/mse_ssim_256x512_lo_200k ckpts/mse_ssim_256x512_mi_200k ckpts/mse_ssim_256x512_hi_200k --out_dir results/SSIM_256x512_lo results/SSIM_256x512_mi results/SSIM_256x512_hi --group SSIM256x512 SSIM256x512 SSIM256x512 --local_image_dir ../CTC-360-resized --results_csv results/resultados_finais_256x512.csv

python -m hific.evaluate --config mselpips_lo mselpips_mi --ckpt_dir ckpts/SWHDC_WSmse_WSssim_256x512_lo_1M ckpts/SWHDC_WSmse_WSssim_256x512_mi_1M --out_dir results/SWHDC_256x512_1M_lo results/SWHDC_256x512_1M_mi --group SWHDC256x512Final SWHDC256x512Final --local_image_dir ../CTC-360-resized --results_csv results/resultados_modelo_finao23.csv

python plot_results.py results/resultados_finais.csv --output results/graficos_comparativos.png

python plot_results.py results/resultados_finais_256x512.csv --output results/graficos_comparativos2.png

DOCKER:

Buildar
SÓ USAR ESSE SE QUISER CRIAR NOVA IMAGEM -> docker build -t hific-360-env .

Rodar
docker run --gpus all -it --rm -v "$(pwd)":/app hific-360-env


target low = 0.14
target mid = 0.3
target high = 0.45

./train_all_models.sh

SWHDC2 - WS SSIM, learn = True, crop 256x512

SWHDC3 - WS SSIM, learn = False, crop 256x512

SWHDC4 - WS SSIM, learn = True, crop 256x256


python -m hific.evaluate --config mselpips_mi --ckpt_dir ckpts/SWHDC6_teste_mi_100k --out_dir results/SWHDC6_teste --group SWHDC6 --local_image_dir .
./CTC-360-resized --results_csv results/resultados_swhdc6_teste.csv


python -m hific.evaluate --config mselpips_lo --ckpt_dir ckpts/SWHDC_WSmse_WSssim_256x512_lo_1M --out_dir results/SWHDC_256x512_1M_lo -
-group SWHDC256x512Final --local_image_dir ../CTC-360-resized --results_csv results/resultados_modelo_finao.csv

% Template for ICIP-2026 paper; to be used with:
%          spconf.sty  - ICASSP/ICIP LaTeX style file, and
%          IEEEbib.bst - IEEE bibliography style file.
% --------------------------------------------------------------------------
\documentclass{article}
\usepackage{spconf,amsmath,graphicx}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{microtype}
\usepackage{caption}
\usepackage{bm}


\usepackage{xcolor}

\newcommand{\todo}[1]{{\color{orange}#1}}
\newcommand{\tlts}[1]{{\color{red}#1}}
\newcommand{\baggio}[1]{{\color{blue}#1}}

% Example definitions.
% --------------------
\def\x{{\mathbf x}}
\def\L{{\cal L}}

% Title.
% ------
\title{HiFiC360: Adapting High-Fidelity Compression for 
%Equirectangular
\tlts{360\textdegree{}}
Images}
%
% Single address.
% ---------------
\name{Author(s) Name(s)\thanks{Thanks to XYZ agency for funding.}}
\address{Author Affiliation(s)}
%
% For example:
% ------------
%\address{School\\
%	Department\\
%	Address}
%
% Two addresses (uncomment and modify for two-address case).
% ----------------------------------------------------------
%\twoauthors
%  {A. Author-one, B. Author-two\sthanks{Thanks to XYZ agency for funding.}}
%	{School A-B\\
%	Department A-B\\
%	Address A-B}
%  {C. Author-three, D. Author-four\sthanks{The fourth author performed the work
%	while at ...}}
%	{School C-D\\
%	Department C-D\\
%	Address C-D}
%
\begin{document}
%\ninept
%
\maketitle
%
\begin{abstract}
%Omnidirectional (360\textdegree{}) 
\tlts{360\textdegree{}}
images\tlts{, which are defined on a spherical surface, are} becoming increasingly popular in virtual reality and immersive applications. 
The \tlts{standard planar representation---the} equirectangular projection format\tlts{---}, however, introduces severe distortions, especially near the poles, and the visually salient equatorial region demands higher fidelity. 
In this paper, we  
\tlts{propose to}
adapt
%the generative image compression model HiFiC (High-Fidelity Image Compression)
\tlts{the High-Fidelity Image Compression (HiFiC) model to properly tackle 360° imagery.}
%to the equirectangular domain, resulting in HiFiC360. 
%We
\tlts{Precisely, we}
investigate three key modifications: (i) replacing the perspective-trained LPIPS loss with 
%structural similarity (SSIM) and its 
%latitude-weighted
\tlts{a weighted-to-spherically-uniform structural similarity}
%variant
(WS-SSIM) \tlts{loss}; (ii) incorporating  
%a spherical-aware
\tlts{spherically-weighted horizontal dilated convolutions}
%convolution
(SWHDC)
%that respects the equirectangular geometry,
\tlts{for feature extraction,}
and (iii) modifying the 
%training
\tlts{image}
crop\tlts{ping} strategy to 
%favor the equatorial region.
\tlts{account for the more informative regions during training.} 
%Models are
\tlts{The model is}
trained on the SUN360 dataset and evaluated 
%on CTC360 using standard and weighted distortion metrics. 
\tlts{under standardized test dataset and metrics.}
%Quantitative results 
\tlts{Results}
show that 
%our best configuration---a 
%fixed geometric weights SWHDC with WS-SSIM loss---achieves substantial improvements in weighted PSNR and weighted SSIM metrics 
\tlts{HiFiC360 achives substantial improvement}
over the baseline,
%HiFIC,
while maintaining competitive bitrates and 
%similar 
training time.
\todo{Compared to the state-of-the-art on 360° image compression, HiFiC360 ...}


\end{abstract}
%
\begin{keywords}
HiFiC,
\tlts{360° Images,}
%360,
%Equirectangular projection, I
\tlts{Neural I}mage \tlts{C}ompression.
\end{keywords}
%
\section{Introduction}
 
%Omnidirectional (360\textdegree{})
\tlts{360\textdegree{}}
images are fundamental for virtual
%reality
(VR)
%,
\tlts{and}
augmented reality (AR), and other immersive applications~\todo{\cite{?}}. 
Captured by specialized cameras with a wide field of view, these images enable fully navigable scenes that conventional photographs cannot represent~\todo{\cite{?}}. 
However, the high spatial resolution required to deliver acceptable perceptual quality across the entire sphere results in very large file sizes, placing heavy demands on storage and transmission bandwidth~\todo{\cite{?}}. Efficient compression is therefore essential to make omnidirectional content practical at scale.
 
The dominant format for representing
%omnidirectional images
\tlts{360\textdegree{} imagery}
is the equirectangular projection (ERP), which maps the spherical surface
\tlts{information}
onto a 
%standard 2D 
rectangle~\todo{\cite{?}}. Although simple to store and process, ERP introduces significant geometric distortions~\todo{\cite{?}}\tlts{. R}egions near the poles are heavily oversampled, while the equatorial band---the region that most often corresponds to the viewer's forward-facing viewport and thus of greatest perceptual importance---is represented with relatively uniform fidelity~\todo{\cite{?}}. 
 
Learned image compression has recently demonstrated a remarkable rate-distortion (R-D) performance, surpassing classical codecs on natural images~\todo{\cite{?}}. Among these approaches, 
%HiFiC
\tlts{High-Fidelity Image Compression (HiFiC)}~\todo{\cite{?}}
achieves particularly high perceptual quality by combining an auto-encoder with a hyperprior entropy model and a patch-based 
%GAN
\tlts{generative-adversarial network (GAN)}
discriminator, using a perceptual loss based on 
%LPIPS
\tlts{Learned Perceptual Image Patch Similarity (LPIPS)}~\todo{\cite{?}}
alongside mean squared error (MSE). 
However, HiFiC was designed and trained exclusively for conventional perspective images. 
%
Two of its core components are therefore mismatched to the equirectangular domain: (i) the LPIPS metric is derived from features of networks trained on perspective 
%photographs
\tlts{imagery}
and does not account for the latitude-dependent distortions present in ERP content; and (ii) standard 2D convolutions implicitly assume a flat image geometry, ignoring the wrap-around continuity at the left and right image boundaries and the stretching \tlts{effect} near the poles.

\todo{Precisamos comentar sobre o SOTA em 360 NIC. Temos que posicionar nosso approach com relação a isso e dizer o porquê de nossa proposta, dado o SOTA existente.}

\todo{Acho que podemos ser mais breves nessa descrição aqui. E deixar detalhes para adiante.}
In this work, we present HiFiC360, a systematic adaptation of HiFiC to the equirectangular domain. We conduct an ablation study in which we progressively introduce three modifications. First, we replace the LPIPS perceptual loss with the structural similarity index (SSIM) and, subsequently, with latitude-weighted variants: WS-MSE and WS-SSIM, which assign greater importance to the perceptually critical equatorial region. Second, we incorporate Sphere-Wrapped Horizontal Deformable Convolutions (SWHDC) --- with both fixed geometric weights and a learnable variant --- into the encoder and decoder to better respect the spherical geometry of the input. Third, we investigate changing the crop strategy. Since the training images are $1024\times512$ pixels, we experiment with $256\times512$ crops (width$\times$height), which span the full latitude of the image from pole to pole, enabling the network to observe the complete latitudinal context at each training step --- a property that proved especially beneficial for the spherically-aware convolutions.

\tlts{The rest of this paper is organized as follows. Section~\ref{sec_related} discusses the related work.}  \todo{Section~...}

\section{Related Works}\label{sec_related}

\todo{TODO: RELATED WORKS SECTION}
\todo{
Sugiro comentar sobre:
\begin{itemize}
\item 10.1109/TIP.2024.3477356 (NIC360)
\item 10.1109/TIP.2022.3208429 (NIC360)
\item 10.1109/TIP.2022.3202357 (NIC360)
\item 10.1109/ICASSP49660.2025.10889131 (NIC360)
\item http://arxiv.org/abs/2402.08862 (NIC360)
\item 10.1007/s11042-025-20876-1 (graph)
\item 10.1109/PCS.2016.7906402 (traditional)
\item https://doi.org/10.1109/LASCAS64004.2025.10966281 (traditional)
\end{itemize}
}

\todo{Movi Datasets para adiante.}

 
%% ============================================================
\section{Methodology}
%% ============================================================

\todo{Incluir um preambulo apresentando essa seção.}
 
\subsection{HiFiC Baseline}
 
HiFiC is a learned generative image compression framework built on a convolutional auto-encoder with a hierarchical (hyperprior) entropy model. The encoder $f_e$ maps an input image $\mathbf{x}$ to a latent representation $\mathbf{y} = f_e(\mathbf{x})$. A hyperprior analysis network produces side information $\mathbf{z}$, from which scale parameters $\hat{\boldsymbol{\sigma}}$ and mean parameters $\hat{\boldsymbol{\mu}}$ are synthesized to model the latent distribution as a conditional Gaussian. Both $\mathbf{y}$ (quantized) and $\mathbf{z}$ (factorized prior) are entropy coded to produce the bitstream. The decoder $f_d$ reconstructs $\hat{\mathbf{x}}$ from the quantized latents. The original paper evaluates two model families: a baseline non-GAN variant trained with an MSE + LPIPS loss, and a GAN variant that adds a patch-discriminator adversarial loss for enhanced perceptual quality at low bitrates. The joint rate--distortion--perception loss is:
\begin{equation}
    \mathcal{L} = R(\mathbf{y}, \mathbf{z}) + \lambda_D \cdot D(\mathbf{x}, \hat{\mathbf{x}}) + \lambda_P \cdot \text{LPIPS}(\mathbf{x}, \hat{\mathbf{x}}),
\end{equation}
where $R$ is the estimated bitrate and $D$ is the MSE distortion. 

\begin{table*}[ht]
\centering
\caption{Bj{\o}ntegaard-delta gains over the MSE+LPIPS baseline (200k steps). Metrics prefixed with \emph{ws-} are Weighted-to-Spherically variants. Negative $\Delta$MSE and $\Delta$WS-MSE indicate improvement. Best weighted-metric values are \textbf{bold}.}
\label{tab:bd}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{Configuration} & $\Delta$PSNR & $\Delta$SSIM & $\Delta$MSE & $\Delta$WS-PSNR & $\Delta$WS-SSIM & $\Delta$WS-MSE \\
 & (dB) & & & (dB) & & \\
\midrule
MSE+SSIM / std.\ conv. / $256\!\times\!256$ & +0.15 & +0.0217 & $-$3.74 & +0.16 & +0.0231 & $-$4.41 \\
WS-MSE+WS-SSIM / std.\ conv. / $256\!\times\!256$ & +0.22 & +0.0224 & $-$4.85 & +0.26 & +0.0252 & $-$6.77 \\
WS-MSE+WS-SSIM / SWHDC / $256\!\times\!256$ & +0.21 & +0.0210 & $-$4.75 & +0.29 & +0.0252 & $-$7.54 \\
MSE+SSIM / std.\ conv. / $256\!\times\!512$ & +0.08 & +0.0277 & $-$3.97 & +0.17 & +0.0295 & $-$5.49 \\
WS-MSE+WS-SSIM / std.\ conv. / $256\!\times\!512$ & +0.08 & +0.0200 & $-$1.54 & +0.41 & +0.0288 & $-$10.63 \\
\textbf{WS-MSE+WS-SSIM / SWHDC / $\bm{256\!\times\!512}$} & +0.17 & +0.0240 & $-$3.85 & \textbf{+0.78} & \textbf{+0.0394} & \textbf{$-$16.42} \\
WS-MSE+WS-SSIM / SWHDC-L / $256\!\times\!512$ & +0.22 & +0.0269 & $-$3.79 & +0.69 & +0.0384 & $-$14.88 \\
\bottomrule
\end{tabular}
\end{table*}
 
\subsection{Loss Function Modifications}
 
As a first ablation step, we replace LPIPS with the Structural Similarity Index Measure (SSIM). SSIM compares images along three perceptually motivated components: luminance $l(\mathbf{x},\hat{\mathbf{x}})$, contrast $c(\mathbf{x},\hat{\mathbf{x}})$, and structure $s(\mathbf{x},\hat{\mathbf{x}})$, computed over local Gaussian-weighted windows of size $11\times11$:
\begin{equation}
    \text{SSIM}(\mathbf{x},\hat{\mathbf{x}}) = l \cdot c \cdot s = \frac{(2\mu_x\mu_{\hat{x}}+C_1)(2\sigma_{x\hat{x}}+C_2)}{(\mu_x^2+\mu_{\hat{x}}^2+C_1)(\sigma_x^2+\sigma_{\hat{x}}^2+C_2)}.
\end{equation}
The distortion term becomes $\mathcal{L}_D = \text{MSE} + \lambda_S(1 - \text{SSIM})$. Unlike LPIPS, SSIM does not rely on features from perspective-trained networks, making it a more domain-neutral perceptual metric for ERP content.
 
In the second ablation step, we replace both loss terms with their latitude-weighted, WS (Weighted-to-Spherically), counterparts. The weight assigned to pixel row $i$ in an image of height $H$ and width $W$ is proportional to the solid angle of the corresponding latitude band:
\begin{equation}
    w_i = \Delta\theta \cdot (\cos\phi_i - \cos\phi_{i+1}), \quad \phi_i = \frac{i\pi}{H},
\end{equation}
where $\Delta\theta = 2\pi/W$. These weights downscale the contribution of polar regions (which are visually less salient and geometrically over-sampled) and up-weight the equatorial band. WS-MSE substitutes the standard mean with a weighted mean over pixel errors, while WS-SSIM computes a weighted average of the per-pixel SSIM map. This formulation directly aligns the training objective with the evaluation metrics used in 360\textdegree{} image quality assessment.
 
\subsection{Spherical Convolution (SWHDC)}
 
Standard 2D convolutions treat the image plane as flat and periodic boundary conditions are not applied, meaning that the horizontal wrap-around of the ERP format (the left and right edges both correspond to the same meridian, and, therefore, are connected) is ignored. To address this, we replace the first and last convolutions in the encoder and decoder --- as well as all convolutions within the residual blocks --- with Sphere-Wrapped Horizontal Deformable Convolutions (SWHDC).
 
SWHDC applies circular padding along the width axis (longitude) to ensure continuity at the image boundary, and symmetric padding at the top and bottom boundaries (the poles) to avoid artificial discontinuities. The convolution is implemented as a weighted sum of dilated convolutions with dilation rates $\{1, 2, 3\}$, where the per-row mixing weights reflect the local sphere-to-plane sampling ratio:
\begin{equation}
    \text{Rs}(\phi_i) = \min\!\left(N,\, \frac{1}{\sin\phi_i}\right),
\end{equation}
with $N$ the number of dilation branches. Rows near the equator ($\phi \approx \pi/2$, $\sin\phi \approx 1$) receive a weight profile concentrated on finer dilation rates, while polar rows receive coarser dilations consistent with the geometric over-sampling. We also evaluate a learnable variant (SWHDC-L) in which the per-row weight profiles are parameterized by a trainable profile of shape $(1, 128, N)$ interpolated to the input height and normalized via softmax, allowing the network to adapt the blending of dilation rates from data rather than fixing them geometrically.

\subsection{Crop Strategy}
\label{sec:crop}
 
The original HiFiC pipeline extracts $256\times256$ crops at uniformly random positions. We experiment with rectangular crops of size $256\times512$ (width $\times$ height). Since the training images are $1024\times512$ pixels, this crop height spans the full vertical extent of the image --- from the south pole to the north pole. Exposing the full latitudinal context at each training step allows the model to observe the complete spectrum of ERP distortions simultaneously, which is particularly beneficial for SWHDC: the spherical weight profile depends on the absolute latitude of each row, and providing the full height ensures that the geometric or learned weights are calibrated against the correct latitude range during training.

\subsection{Training and Evaluation Protocol}
\label{sec:training}

All models are optimized with Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$) with an initial learning rate of $10^{-4}$, decaying by a factor of $0.1$ after $500{,}000$ steps. To explore the large configuration space --- spanning loss functions, convolution types, and crop strategies ---,  all candidate configurations are first trained as toy models for $200{,}000$ steps with a batch size of 8, using the non-GAN backbone. As a reference, we also train a baseline model under the same $200{,}000$-step budget using the original MSE + LPIPS loss and standard convolutions, ensuring a fair comparison that is not confounded by training length.
 
For each configuration, three rate-point variants are trained following the HiFiC bitrate targets: \texttt{lo} ($r_t = 0.14$ bpp), \texttt{mi} ($r_t = 0.30$ bpp), and \texttt{hi} ($r_t = 0.45$ bpp). The toy models are compared against the 200k-step baseline using Bj{\o}ntegaard-delta (BD) analysis computed over the WS-PSNR, WS-SSIM, and WS-MSE metrics across the three rate points. The configuration achieving the best BD gains on the weighted spherical metrics is selected for a full training of $1{,}000{,}000$ steps (for reference, the main HiFiC model is trained for $2{,}000{,}000$ steps).
 
Then, the resulting final model is also evaluated on the CTC360 dataset alongside other reference models, enabling a direct comparison on equirectangular content under consistent evaluation conditions. GAN-based variants were not pursued, as preliminary experiments showed that the non-GAN models achieved superior performance on the weighted spherical metrics that form our primary evaluation criteria.

%% ============================================================
\section{Results and Analysis}
\label{sec:results}
%% ============================================================


%% ============================================================
\subsection{Datasets}
%% ============================================================
\todo{Movi Datasets para cá.}

\subsubsection{SUN360 (Training Set)}
 
We train all models on the SUN360 dataset, a large-scale collection of $48{,}319$ equirectangular images at $1024\times512$ pixel resolution. The dataset spans a broad range of scene categories, including indoor environments (offices, living rooms, corridors), outdoor landscapes, and urban streetscapes. This scene diversity is valuable for training a compression model, as it exposes the network to a wide distribution of textures, luminance conditions, and structural patterns, promoting generalization.
 
\subsubsection{CTC360 (Evaluation Set)}
 
For evaluation, we use the CTC360 dataset, which consists of 30 high-quality equirectangular images comprising indoor scenes and urban environments. The images in CTC360 were originally captured at varying resolutions. To ensure a fair and consistent comparison across all models, we resized each image to $1024\times512$ pixels using Lanczos resampling, which is well suited to downscaling as it minimizes aliasing while preserving fine detail.


\subsection{Evaluation Metrics}
 
We evaluate all models on the CTC360 resized dataset using six complementary metrics. The standard metrics --- PSNR, SSIM, and MSE --- provide a reference comparison with the conventional image compression literature. However, because these metrics assign equal importance to all pixels regardless of their position in the equirectangular projection, they do not reflect the actual perceptual experience of a human viewer, who predominantly fixates on equatorial regions.
 
We therefore place emphasis on the Weighted-to-Spherically variants: WS-PSNR, WS-SSIM, and WS-MSE. We argue that improvements on these metrics translate more directly into better perceptual quality for real-world omnidirectional viewing. All reported values on each of these metrics represent averages across the 30 images of the CTC360 evaluation set, calculated separately for each of the three rate-point models.
\begin{figure*}[ht]
    \centering
    \includegraphics[width=0.85\textwidth]{graficos_comparativos2.png}
    \caption{Exemplo de gráfico comparando nosso modelo final com outros}
\end{figure*}
 
\subsection{Ablation Study Results}

Table~\ref{tab:bd} reports the BD gains for each configuration against the standard baseline. The results demonstrate that all proposed modifications individually improve weighted spherical metrics. However, simply increasing crop size to $256\!\times\!512$ with standard convolutions yields marginal or even negative gains on standard metrics compared to square crops. The true benefit of full-latitude crops emerges when paired with spherically-aware convolutions. The synergy of the weighted loss, SWHDC, and full-latitude crops (WS-MSE+WS-SSIM / SWHDC / $256\!\times\!512$) achieved the highest overall improvements (+0.78 dB WS-PSNR, +0.0394 WS-SSIM, $-$16.42 WS-MSE). Notably, the learnable-weight SWHDC-L variant also produced slightly lower, but still competitive results ($+0.69$ dB, $+0.0384$, $-14.88$). Accordingly, we selected the fixed-weight SWHDC configuration for the final 1M-step training.
 
\subsection{Final Model Results}


TODO: FINAL MODEL ANALYSIS WITH OTHER EXTERNAL MODELS (INCLUDING BASE HIFIC)
 
%% ============================================================
\section{Conclusion and Future Work}
%% ============================================================
 
We presented HiFiC360, a systematic adaptation of the HiFiC generative compression framework to the equirectangular domain. Through a structured ablation study, we demonstrated that replacing the perspective LPIPS loss with latitude-weighted objectives (WS-MSE + WS-SSIM) consistently improves compression quality in the equatorial region --- the perceptually most important part of the 360\textdegree{} image sphere. Incorporating SWHDC convolutions, which enforce horizontal wrap-around continuity and account for the latitude-dependent sampling density of the ERP format, yields further gains, particularly when combined with $256\times512$ training crops that expose the network to the full latitudinal range. Our best configuration achieves substantial improvements in WS-PSNR and WS-SSIM metrics over the HiFiC baseline while operating at comparable bitrates.
 
Several directions remain open for future work. An immediate next step is to evaluate generalization across other 360\textdegree{} datasets and extend training beyond the current schedule --- aligning it with the more extensive training budgets (e.g., 2M or 6M steps) reported in related literature. Additionally, all experiments here used the non-GAN variant of HiFiC; it would be valuable to further investigate whether our modifications yield improvements when combined with the adversarial discriminator. Finally, a similar adaptation pipeline could be applied to alternative omnidirectional formats, such as cubemap or equi-angular cubemap projections, and to other learned compression architectures.

%The SWHDC-L learnable variant could also benefit from longer training or a more expressive parameterization, as the fixed geometric prior may become a bottleneck at higher model capacity

\bibliographystyle{IEEEbib}
\bibliography{strings,refs}

\end{document}
