# Antibody target classification from sequence information

Monoclonal antibodies are highly specific therapeutic options for treating various diseases. However, the traditional approach to developing neutralizing antibodies has limitations, including a lack of control over target selection. Improving the screening strategy is crucial, and identifying the target molecule's molecular nature can prioritize potential candidates for in-vitro characterization. In this work, we explore the use of deep learning algorithms for identifying the target nature of monoclonal antibodies based on sequence information.

# Description
* train.py: training of ProtBERT and AntiBERTy model using MLP
* train_OCSVM.py: training of ProtBERT and AntiBERTy model using OCSVM
* pred.py: inference on ProtBERT and AntiBERTy
* pred_ensamble.py: inference using the ensamble models on ProtBERT and AntiBERty
