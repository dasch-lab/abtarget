# Antibody target classification from sequence information

Monoclonal antibodies are highly specific therapeutic options for treating various diseases. However, the traditional approach to developing neutralizing antibodies has limitations, including a lack of control over target selection. Improving the screening strategy is crucial, and identifying the target molecule's molecular nature can prioritize potential candidates for in-vitro characterization. In this work, we explore the use of deep learning algorithms for identifying the target nature of monoclonal antibodies based on sequence information.

# Description
* train_bootstrap.py: training and evaluation of ProtBERT and AntiBERTy models using MLP with the opportunity to perform also the bootstrap analysis
* train_OCSVM.py: training and evaluation of ProtBERT and AntiBERTy models using OCSVM
* pred.py: ensemble of ProtBERT and AntiBERty for all the experiments except SMOTE augmentation
* pred_SMOTE.py: ensemble of ProtBERT and AntiBERty for SMOTE augmentation
