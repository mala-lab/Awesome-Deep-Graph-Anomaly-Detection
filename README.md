# Awesome-Deep-Graph-Anomaly-Detection
A repository for resources of deep learning-based graph anomaly detection.
We have categorized the literature related to deep graph anomaly detection according to the taxonomies of methodology design. For more details, please refer to our survey.


## 1. Graph Neural Network Design 
#### Aggregation Mechanism
| CARE-GNN  | Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters  | CIKM 2020  | [PDF](https://penghao-bdsc.github.io/papers/cikm20.pdf)  | [PyTorch](https://github.com/YingtongDou/CARE-GNN)  |
| GraphConsis  | Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection  | SIGIR 2020  | [PDF](https://par.nsf.gov/servlets/purl/10167818)  | [TensorFlow](https://github.com/safe-graph/DGFraud)  |
| GDN  | Graph Neural Network-Based Anomaly Detection in Multivariate Time Series  | AAAI 2021  | [PDF](https://cdn.aaai.org/ojs/16523/16523-13-20017-1-2-20210518.pdf)  | [PyTorch](https://github.com/d-ailin/GDN)  |
| PC-GNN  | Pick and choose: a GNN-based imbalanced learning approach for fraud detection  | WWW 2021  | [PDF](https://ponderly.github.io/pub/PCGNN_WWW2021.pdf)  | [PyTorch](https://github.com/PonderLY/PC-GNN)  |
| GHRN  | Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum  | WWW 2023  | [PDF](https://hexiangnan.github.io/papers/www23-graphAD.pdf)  | [PyTorch](https://github.com/blacksingular/GHRN)  |
| H2-FDetector  | H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections  | WWW 2022  | [PDF](https://scholar.archive.org/work/fomltdkxnrblndckrapxjyusri/access/wayback/https://dl.acm.org/doi/pdf/10.1145/3485447.3512195)  | [PyTorch](https://github.com/shifengzhao/H2-FDetector)  |
| AO-GNN  | AUC-oriented Graph Neural Network for Fraud Detection  | WWW 2022  | [PDF](https://ponderly.github.io/pub/AOGNN_WWW2022.pdf)  | [N/A]  |
| BWGNN  | Rethinking Graph Neural Networks for Anomaly Detection  | ICML 2022  | [PDF](https://www.researchgate.net/profile/Jia-Li-127/publication/360994234_Rethinking_Graph_Neural_Networks_for_Anomaly_Detection/links/6299d59b6886635d5cbb9bb1/Rethinking-Graph-Neural-Networks-for-Anomaly-Detection.pdf)  | [PyTorch](https://github.com/squareroot3/rethinking-anomaly-detection)  |
| FRAUDRE  | FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance  | ICDM 2021  | [PDF](https://www.researchgate.net/profile/Chuan-Zhou-3/publication/357512222_FRAUDRE_Fraud_Detection_Dual-Resistant_to_Graph_Inconsistency_and_Imbalance/links/61d18807b8305f7c4b19bd14/FRAUDRE-Fraud-Detection-Dual-Resistant-to-Graph-Inconsistency-and-Imbalance.pdf)  | [PyTorch](https://github.com/FraudDetection/FRAUDRE)  |
| DCI  | Decoupling Representation Learning and Classification for GNN-based Anomaly Detection  | SIGIR 2021  | [PDF](https://xiaojingzi.github.io/publications/SIGIR21-Wang-et-al-decoupled-GNN.pdf)  | [PyTorch](https://github.com/wyl7/DCI-pytorch)  |


#### Distinguishable Feature Extraction
| DAGAD  | DAGAD: Data Augmentation for Graph Anomaly Detection  | ICDM 2022  | [PDF](https://ieeexplore.ieee.org/abstract/document/10027747/)  | [PyTorch](https://github.com/fanzhenliu/dagad)  |
| GDN  | Alleviating Structural Distribution Shift in Graph Anomaly Detection  | WSDM 2023  | [PDF](https://hexiangnan.github.io/papers/wsdm23-GDN.pdf)  | [PyTorch](https://github.com/blacksingular/wsdm_GDN)  |


## 2.Self-Supervised Proxy Task Design

#### Contrastive Learning


#### Property tailored Task


| HCM  | Hop-Count Based Self-Supervised Anomaly Detection on Attributed Networks  | ECML PKDD 2022  | [PDF](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_927.pdf)  | [PyTorch](https://github.com/TienjinHuang/GraphAnomalyDetection)  |

#### Reconstruction-based Feature Learning
| Dominant| Deep Anomaly Detection on Attributed Networks|SDM2019 |[PDF]([https://ieeexplore.ieee.org/abstract/document/9395172/](http://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf))  | [PyTorch](https://github.com/kaize0409/GCN_AnomalyDetection/blob/master/)  |
| CoLA  | Anomaly detection on attributed networks via contrastive self-supervised learning  | TNNLS 2021  | [PDF](https://ieeexplore.ieee.org/abstract/document/9395172/)  | [PyTorch](https://github.com/grand-lab/cola)  |
| ComGA  | ComGA: Community-Aware Attributed Graph Anomaly Detection  | WSDM 2022  | [PDF](https://dl.acm.org/doi/abs/10.1145/3488560.3498389)  | [TensorFlow](https://github.com/XuexiongLuoMQ/ComGA)  |


## 3. Anomaly Measures
#### One Class Distance/Affinity Measure
| DeepSphere  | Deep into Hypersphere: Robust and Unsupervised Anomaly Discovery in Dynamic Networks  | IJCAI 2018  | [PDF](https://www.ijcai.org/Proceedings/2018/0378.pdf)  | [TensorFlow](https://github.com/picsolab/DeepSphere)  |
#### Knowledge Distillation
| GLocalKD  | Deep Graph-level Anomaly Detection by Glocal Knowledge Distillation  | WSDM 2022  | [PDF](https://arxiv.org/abs/2112.10063)  | [PyTorch](https://github.com/RongrongMa/GLocalKD)  |

#### Prior distribution-driven Models
#### Adverisrial Learning Scoring

| GAAN  | Generative Adversarial Attributed Network Anomaly Detection  | CIKM 2020  | [PDF](https://static.aminer.cn/storage/pdf/acm/20/cikm/10.1145/3340531.3412070.pdf)  | [PyTorch](https://github.com/Kaslanarian/SAGOD)  |
