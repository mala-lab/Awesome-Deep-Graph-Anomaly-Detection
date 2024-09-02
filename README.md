# Deep Graph Anomaly Detection (DGAD): A Survey and New Perspectives
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/mala-lab/Awesome-Deep-Graph-Anomaly-Detection)
[![Visits Badge](https://badges.pufler.dev/visits/mala-lab/Awesome-Deep-Graph-Anomaly-Detection)](https://badges.pufler.dev/visits/mala-lab/Awesome-Deep-Graph-Anomaly-Detection)
<!-- ![Forks](https://img.shields.io/github/forks/mala-lab/Awesome-Deep-Graph-Anomaly-Detection/) -->

A professionally curated list of awesome resources (paper, code, data, etc.) on **Deep Graph Anomaly Detection (DGAD)**, which is the first work to comprehensively and systematically summarize the recent advances of deep graph anomaly detection from the methodology design to the best of our knowledge.

We will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.

 
## Survey Paper 

[**Deep Graph Anomaly Detection: A Survey and New Perspectives**](Arxiv)  

[Hezhe Qiao](https://hezheqiao2022.github.io/), [Hanghang Tong](http://tonghanghang.org/), [Bo An](https://personal.ntu.edu.sg/boan/), [Irwin King](https://www.cse.cuhk.edu.hk/people/faculty/irwin-king/), [Charu Aggarwal](https://www.charuaggarwal.net/), [Guansong Pang](https://sites.google.com/site/gspangsite/home).


[****](Arxiv)  



#### If you find this repository helpful for your work, please kindly cite our paper.

```bibtex

```


## Taxonomy of  Deep Graph Anomaly Detection
<img src="overview.png" width=900 align=middle> <br />

<!-- ![xxx](generative_adversarial_ssl4ts.jpg) -->
<img src="Fig1.png" width=900 align=middle> <br />

<!-- ![xxx](generative_adversarial_ssl4ts.jpg) -->
<img src="Fig2.png" width=900 align=middle> <br />

<!-- ![xxx](generative_adversarial_ssl4ts.jpg) -->
<img src="Fig3.png" width=850 align=middle> <br />



# Category of Deep Graph Anomaly Detection

## GNN Backbone Design 

### Discriminative GNNs

#### Aggregation Mechanism

- Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters in *CIKM*, 2020. [\[paper\]](https://arxiv.org/abs/2008.08692)[\[code\]](https://github.com/YingtongDou/CARE-GNN)

- Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection in *SIGIR*, 2020. [\[paper\]](https://arxiv.org/abs/2005.00625)[\[code\]](https://github.com/safe-graph/DGFraud)

- Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection in *WWW*, 2021.[\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3442381.3449989)[\[code\]](https://github.com/PonderLY/PC-GNN)

- FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance  in *ICDM*, 2021. [\[paper\]](https://ieeexplore.ieee.org/document/9679178)[\[code\]](https://github.com/FraudDetection/FRAUDRE)

- Dual-discriminative Graph Neural Network for Imbalanced Graph-level Anomaly Detection in *NeurIPS*, 2022. [\[paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/98a625423070cfc6ae3d82d4b59408a0-Abstract-Conference.html)

- Explainable Graph-based Fraud Detection via Neural Meta-graph Search in *CIKM*, 2022. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3511808.3557598)[\[code\]](https://github.com/qzzdd/NGS)

- Bi-Level Selection via Meta Gradient for Graph-based Fraud Detection in *DASFAA *, 2022. [\[paper\]](https://yliu.site/pub/BLS_DASFAA2022.pdf)

- H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections in *WWW*, 2022. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3485447.3512195)

- Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum in *WWW*, 2023. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3543507.3583268)[\[code\]](https://github.com/blacksingular/GHRN)


- Towards Graph-level Anomaly Detection via Deep Evolutionary Mapping in *KDD*, 2023. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599524)[\[code\]](https://github.com/XiaoxiaoMa-MQ/GmapAD/)

- Multitask Active Learning for Graph Anomaly Detection in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2401.13210)[\[code\]](https://github.com/AhaChang/MITIGATE)

- Generation is better than Modification: Combating High Class Homophily Variance in Graph Anomaly Detection in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2403.10339)

- Boosting Graph Anomaly Detection with Adaptive Message Passing in *ICLR*, 2024. [\[paper\]](https://openreview.net/forum?id=CanomFZssu)

-  Partitioning Message Passing for Graph Fraud Detection in *ICLR*, 2024. [\[paper\]](https://openreview.net/pdf?id=tEgrUrUuwA)[\[code\]](https://github.com/Xtra-Computing/PMP)

- Graph Anomaly Detection with Bi-level Optimization in *WebConf*, 2024. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3589334.3645673)[\[code\]](https://github.com/blacksingular/Bio-GNN)


#### Feature Transformation

- Can Abnormality be Detected by Graph Neural Networks? in *IJCAI*, 2022. [\[paper\]](http://yangy.org/works/gnn/IJCAI22_Abnormality.pdf)[\[code\]](https://github.com/zjunet/AMNet)

- Rethinking Graph Neural Networks for Anomaly Detection in *ICML*, 2022.[\[paper\]](https://proceedings.mlr.press/v162/tang22b.html)[\[code\]](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)

- Alleviating Structural Distribution Shift in Graph Anomaly Detection in *WSDM*, 2023.[\[paper\]](https://arxiv.org/abs/2401.14155)[\[code\]](https://github.com/blacksingular/wsdm_GDN)

- Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection in *ICLR*, 2024.[\[paper\]](https://arxiv.org/pdf/2310.02861)[\[code\]](https://github.com/xydong127/RQGNN)

- SmoothGNN: Smoothing-based GNN for Unsupervised Node Anomaly Detection in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2405.17525)



### Generative GNNs

#### Feature Interpolation


- GRAPHENS:Neighbor-aware Ego Network Synthesis for Class-imbalance Node Classification in *ICLR*, 2022. [\[paper\]](https://openreview.net/forum?id=MXEl7i-iru)[\[code\]](https://github.com/JoonHyung-Park/GraphENS)

- DAGAD: Data Augmentation for Graph Anomaly Detection in *ICDM*, 2022. [\[paper\]](https://arxiv.org/abs/2210.09766)[\[code\]](https://github.com/FanzhenLiu/DAGAD)

- GADY Unsupervised Anomaly Detection on Dynamic Graphs in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2310.16376)

- Generative Graph Augmentation for Minority Class in Fraud Detection in *CIKM*, 2023. [\[paper\]](https://dl.acm.org/doi/10.1145/3583780.3615255)

- Improving Generalizability of Graph Anomaly Detection Models via Data Augmentation in *TKDE*, 2023. [\[paper\]](https://arxiv.org/abs/2306.10534v1)[\[code\]](https://github.com/betterzhou/AugAN)

- Class-Imbalanced Graph Learning without Class Rebalancing in *ICML*, 2024. [\[paper\]](https://arxiv.org/abs/2308.14181)[\[code\]](https://github.com/ZhiningLiu1998/BAT)

- Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision in *ICLR*, 2024. [\[paper\]](https://openreview.net/pdf?id=elMKXvhhQ9)[\[code\]](https://github.com/Xtra-Computing/ConsisGAD)

#### Noise Perturbation


- Self-Discriminative Modeling for Anomalous Graph Detection in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2310.06261)

- Generative Semi-supervised Graph Anomaly Detection in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2402.11887)[\[code\]](https://github.com/mala-lab/GGAD)

- GODM Data Augmentation for Supervised Graph Outlier Detection with Latent Diffusion Models in *Arxiv*, 2023.  [\[paper\]](https://arxiv.org/abs/2312.17679)[\[code\]](https://github.com/kayzliu/godm)

- GADY: Unsupervised Anomaly Detection on Dynamic Graphs in *Arxiv*, 2023.  [\[paper\]](https://arxiv.org/abs/2310.16376)[\[code\]](https://github.com/mufeng-74/GADY)

- Graph Anomaly Detection with Few Labels: A Data-Centric Approach in *KDD*, 2024. [\[paper\]](https://dl.acm.org/doi/10.1145/3637528.3671929) 

## Proxy Task Design

### Data Reconstruction 

- NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks in *KDD*, 2018. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3219819.3220024)[\[code\]](https://github.com/chengw07/NetWalk)

- Deep Anomaly Detection on Attributed Networks in *SDM*, 2019. [\[paper\]](https://epubs.siam.org/doi/epdf/10.1137/1.9781611975673.67)[\[code\]](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch)

- ANOMALYDAE: Dual Autoencoder for Anomaly Detection on Attribute Networks in *ICASSP*, 2020. [\[paper\]](https://arxiv.org/abs/2002.03665)[\[code\]](https://github.com/haoyfan/AnomalyDAE)

- Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding in *WSDM*, 2020. [\[paper\]](https://dl.acm.org/doi/10.1145/3336191.3371788)[\[code\]](https://github.com/vasco95/DONE_AdONE)

- ResGCN Attention-based Deep Residual Modeling for Anomaly Detection on Attributed Networks in *Machine Learning*, 2021. [\[paper\]](https://arxiv.org/abs/2009.14738)[\[code\]](https://bitbucket.org/paulpei/resgcn/src/master/)

- Mul-GAD: a semi-supervised graph anomaly detection framework via aggregating multi-view information in *Arxiv*, 2022. [\[paper\]](https://arxiv.org/abs/2212.05478)[\[code\]](https://github.com/liuyishoua/Mul-Graph-Fusion)

- AnomMAN: Detect Anomaly on Multi-view Attributed Networks in *Information Sciences*, 2022.[\[paper\]](https://arxiv.org/abs/2201.02822)

- A Deep Multi-View Framework for Anomaly in *TKDE*, 2022. [\[paper\]](https://ieeexplore.ieee.org/document/9162509)

- ComGA: Community-Aware Attributed Graph Anomaly Detection in *WSDM*, 2022. [\[paper\]](https://dl.acm.org/doi/10.1145/3488560.3498389)[\[code\]](https://github.com/XuexiongLuoMQ/ComGA)

- Reconstruction Enhanced Multi-View Contrastive Learning for Anomaly Detection on Attributed Networks in *IJCAI*, 2022. [\[paper\]](https://arxiv.org/abs/2205.04816)[\[code\]](https://github.com/Zjer12/Sub)

- Unsupervised Graph Outlier Detection: Problem Revisit, New Insight, and Superior Method in *Arxiv*, 2022. [\[paper\]](https://arxiv.org/abs/2210.12941)[\[code\]](https://github.com/goldenNormal/vgod-github)

- Graph-level Anomaly Detection via Hierarchical Memory Networks in *ECML PKDD*, 2023. [\[paper\]](https://arxiv.org/abs/2307.00755)[\[code\]](https://github.com/Niuchx/HimNet)

- Hybrid-Order Anomaly Detection on Attributed Networks in *TKDE*, 2023 [\[paper\]](https://ieeexplore.ieee.org/document/9560054)[\[code\]](https://github.com/zirui-yuan/HO-GAT)

- A graph encoder–decoder network for unsupervised anomaly detection in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2308.07774)

- Label-based Graph Augmentation with Metapath for Graph Anomaly Detection in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2308.10918)[\[code\]](https://github.com/missinghwan/INFOREP)

- GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction in *WSDM*, 2024.  [\[paper\]](https://arxiv.org/abs/2306.01951)[\[code\]](https://github.com/Graph-COM/GAD-NR)

- ADA-GAD:Anomaly-Denoised Autoencoders for Graph Anomaly Detection in *AAAI*, 2024. [\[paper\]](https://arxiv.org/abs/2312.14535)[\[code\]](https://github.com/jweihe/ADA-GAD)

- STRIPE Spatial-temporal Memories Enhanced Graph Autoencoder for Anomaly Detection in Dynamic Graphs in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2403.09039)

### Contrastive Learning

- ANEMONE: Graph Anomaly Detection with Multi-Scale Contrastive Learning in *CIKM*, 2021. [\[paper\]](https://dl.acm.org/doi/10.1145/3459637.3482057)


- Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection in *TKDE*, 2021. [\[paper\]](https://arxiv.org/abs/2108.09896)[\[code\]](https://github.com/KimMeen/SL-GAD)

- Anomaly Detection in Dynamic Graphs via Transformer in *TKDE*, 2021. [\[paper\]](https://arxiv.org/abs/2106.09876)[\[code\]](https://github.com/yuetan031/TADDY_pytorch)

- CoLA Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning in *TNNLS*, 2021. [\[paper\]](https://arxiv.org/abs/2103.00113)[\[code\]](https://github.com/TrustAGI-Lab/CoLA)


- CONDA Contrastive Attributed Network Anomaly Detection with Data Augmentation in *PAKDD*, 2022. [\[paper\]](https://link.springer.com/chapter/10.1007/978-3-031-05936-0_35)[\[code\]](https://github.com/zhiming-xu/conad)

- Decoupling Representation Learning and Classification for GNN-based Anomaly Detection in *SIGIR*, 2021. [\[paper\]](https://dl.acm.org/doi/10.1145/3404835.3462944)[\[code\]](https://github.com/wyl7/DCI-pytorch)

- GCCAD:Graph Contrastive Coding for Anomaly Detection in *TKDE*, 2022. [\[paper\]](https://ieeexplore.ieee.org/document/9870034/)[\[code\]](https://github.com/THUDM/GraphCAD)

- Cross-Domain Graph Anomaly Detection via Anomaly-aware Contrastive Alignment in *AAAI*, 2022. [\[paper\]](https://arxiv.org/abs/2212.01096)[\[code\]](https://github.com/QZ-WANG/ACT)

- Reconstruction Enhanced Multi-View Contrastive Learning for Anomaly Detection on Attributed Networks in *IJCAI*, 2022. [\[paper\]](https://arxiv.org/abs/2205.04816)[\[code\]](https://github.com/Zjer12/Sub)

- Few-shot Message-Enhanced Contrastive Learning for Graph Anomaly Detection  in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2311.10370)

- ARISE: Graph Anomaly Detection on Attributed Networks via Substructure Awareness in *TNNLS*, 2023. [\[paper\]](https://arxiv.org/abs/2211.15255)[\[code\]](https://github.com/FelixDJC/ARISE)

- BOURNE: Bootstrapped Self-supervised Learning Framework for Unified Graph Anomaly Detection in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2307.15244)

- GOOD-D:On Unsupervised Graph Out-Of-Distribution Detection in *WSDM*, 2023.  [\[paper\]](https://arxiv.org/abs/2211.04208)[\[code\]](https://github.com/yixinliu233/g-ood-d)

- GRADATE:Graph Anomaly Detection via Multi-Scale Contrastive Learning Networks with Augmented View in *AAAI*, 2023.  [\[paper\]](https://arxiv.org/abs/2212.00535)[\[code\]](https://github.com/FelixDJC/GRADATE)

- GraphFC:Customs Fraud Detection with Label Scarcity in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2305.11377)[\[code\]](https://github.com/k-s-b/gnn_wco)

- Revisiting Graph Contrastive Learning for Anomaly Detection in *Arxiv*, 2023.   [\[paper\]](https://arxiv.org/abs/2305.02496)[\[code\]](https://github.com/liuyishoua/MAG-Framework)

- Multi-representations Space Separation based Graph-level Anomaly-aware Detection in *SSDBM*, 2023.  [\[paper\]](https://arxiv.org/abs/2307.12994)[\[code\]](https://github.com/whb605/mssgad)

- Towards Self-Interpretable Graph-Level Anomaly Detection in *NeurIPS*, 2023. [\[paper\]](https://arxiv.org/abs/2310.16520)[\[code\]](https://github.com/yixinliu233/signet)

- Learning Node Abnormality with Weak Supervision  in *CIKM*, 2023. [\[paper\]](https://dl.acm.org/doi/10.1145/3583780.3614950)

- Federated Graph Anomaly Detection via Contrastive Self-Supervised Learning in *TNNLS*, 2024. [\[paper\]](https://ieeexplore.ieee.org/document/10566052)


### Knowledge Distillation 

- Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation in *CIKM*, 2020. [\[paper\]](https://dl.acm.org/doi/10.1145/3340531.3412070)[\[code\]](https://git.io/GLocalKD)

- Discriminative Graph-level Anomaly Detection via Dual-students-teacher Model in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2308.01947)

- FGAD: Self-boosted Knowledge Distillation for An Effective Federated Graph Anomaly Detection Framework in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2402.12761)


### Adversarial Score Learning

- Generative Adversarial Attributed Network Anomaly Detection in *CIKM*, 2020. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3340531.3412070)[\[code\]](https://github.com/pygod-team/pygod)

- Inductive Anomaly Detection on Attributed Networks in *IJCAI*, 2021. [\[paper\]](https://www.ijcai.org/proceedings/2020/0179.pdf)[\[code\]](https://github.com/pygod-team/pygod)

- Counterfactual_Graph_Learning_for_Anomaly_Detection_on_Attributed_Networks in *TKDE*, 2023. [\[paper\]](https://github.com/ChunjingXiao/CFAD/blob/main/TKDE_23_CFAD.pdf)[\[code\]](https://github.com/ChunjingXiao/CFAD)

- Generative Graph Augmentation for Minority Class in Fraud Detection  in *Arxiv*, 2023. [\[paper\]](https://dl.acm.org/doi/10.1145/3583780.3615255)[\[code\]](https://github.com/ChunjingXiao/CFAD)

### Score Prediction

- DevNet Deep Anomaly Detection with Deviation Networks in *KDD*, 2019. [\[paper\]](https://arxiv.org/abs/1911.08623)[\[code\]](https://github.com/GuansongPang/deviation-network)

- Few-shot Network Anomaly Detection via Cross-network in *WebConf*, 2021. [\[paper\]](https://arxiv.org/pdf/2102.11165)[\[code\]](https://github.com/kaize0409/Meta-GDN_AnomalyDetection)

- SAD:Semi-Supervised Anomaly Detection on Dynamic Graphs in *IJCAI*, 2023. [\[paper\]](https://arxiv.org/abs/2305.13573)[\[code\]](https://github.com/D10Andy/SAD)

- Learning Node Abnormality with Weak Supervision in *CIKM*, 2023. [\[paper\]](https://dl.acm.org/doi/10.1145/3583780.3614950)



## Graph Anomaly Measure

### One-class Distance

- Deep into Hypersphere: Robust and Unsupervised Anomaly Discovery in Dynamic Networks in *IJCAI*, 2018. [\[paper\]](https://www.ijcai.org/proceedings/2018/0378.pdf)[\[code\]](https://github.com/picsolab/DeepSphere)

- Subtractive Aggregation for Attributed Network Anomaly Detection in *CIKM*, 2021. [\[paper\]](https://dl.acm.org/doi/10.1145/3459637.3482195?cid=99659129036)[\[code\]](https://github.com/betterzhou/AAGNN)

- HRGCN: Heterogeneous Graph-level Anomaly Detection with Hierarchical Relation-augmented Graph Neural Networks in *DSAA*, 2023. [\[paper\]](https://arxiv.org/abs/2308.14340)[\[code\]](https://github.com/jiaxililearn/HRGCN)

- Deep Graph-level Orthogonal Hypersphere Compression for Anomaly Detection in *ICLR*, 2024. [\[paper\]](https://arxiv.org/abs/2302.06430)[\[code\]](https://github.com/wownice333/DOHSC-DO2HSC)


### Community Adherence

- NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks in *KDD*, 2018. [\[paper\]](https://dl.acm.org/doi/10.1145/3219819.3220024)[\[code\]](https://github.com/chengw07/NetWalk)


- Unseen Anomaly Detection on Networks via Multi-Hypersphere Learning in *SDM*, 2024. [\[paper\]](https://epubs.siam.org/doi/10.1137/1.9781611977172.30)[\[code\]](https://github.com/betterzhou/MHGL)



### Local Affinity

- Class Label-aware Graph Anomaly Detection in *CIKM*, 2023. [\[paper\]](https://arxiv.org/abs/2308.11669)[\[code\]](https://github.com/jhkim611/CLAD)

- PREM: A Simple Yet Effective Approach for Node-Level Graph Anomaly Detection in *ICDM*, 2023. [\[paper\]](https://arxiv.org/abs/2310.11676)[\[code\]](https://github.com/CampanulaBells/PREM-GAD)

- Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection in *NeurIPS*, 2023. [\[paper\]](https://arxiv.org/abs/2306.00006)[\[code\]](https://github.com/mala-lab/TAM-master/)

- ARC: A Generalist Graph Anomaly Detector with In-Context Learning in *Arxiv*, 2024.  [\[paper\]](https://arxiv.org/abs/2405.16771)



### Graph Isolation 

- Deep Isolation Forest for Anomaly Detection in *TKDE*, 2023. [\[paper\]](https://arxiv.org/abs/2206.06602)[\[code\]](https://github.com/xuhongzuo/deep-iforest)

- Subgraph Centralization: A Necessary Step for Graph Anomaly Detection in *SDM*,2023.  [\[paper\]](https://arxiv.org/abs/2301.06794)[\[code\]](https://github.com/IsolationKernel/Codes)


## Graph Anomaly Detection Related Survey

- A Comprehensive Survey on Graph Anomaly Detection with Deep Learning in *TKDE*, 2021.  [\[paper\]](https://arxiv.org/abs/2106.07178)

- BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs in *NeurIPS*, 2022. [\[paper\]](https://arxiv.org/abs/2206.10071)[\[code\]](https://github.com/pygod-team/pygod/tree/main/benchmark)

- GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection in *NeurIPS*, 2023. [\[paper\]](https://arxiv.org/abs/2306.12251)[\[code\]](https://github.com/squareRoot3/GADBench)

- Unifying Unsupervised Graph-Level Anomaly Detection and Out-of-Distribution Detection:A Benchmark in *Arxiv*, 2024. [\[paper\]](https://arxiv.org/abs/2406.15523)[\[code\]](https://github.com/UB-GOLD/UB-GOLD)

##  Anomaly Detection Related Survey

- Deep Learning for Anomaly Detection: A Review in *CSUR*, 2020. [\[paper\]](https://arxiv.org/abs/2007.02500)

- Weakly Supervised Anomaly Detection: A Survey in *Arxiv*, 2023. [\[paper\]](https://arxiv.org/abs/2302.04549)[\[code\]](https://github.com/yzhao062/wsad)
