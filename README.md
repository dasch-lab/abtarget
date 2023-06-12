# Antibody target classification from sequence information

Monoclonal antibodies offer highly specific therapeutic options for treating a broad range of diseases. However, the traditional approach to developing neutralizing antibodies has limitations, including a lack of control over target selection. Therefore, improving the screening strategy is of utmost importance. In particular, identifying the molecular nature of the target molecule holds significant value in prioritizing potential candidates for subsequent in-vitro characterization. To address these challenges, the integration of in vitro and in silico methods is necessary. 
In this article, we explore the use of deep learning algorithms for identifying the target nature of monoclonal antibodies based on sequence information, highlighting the importance of this approach for accelerating and optimizing the development process. To the best of our knowledge, this study represents the initial attempt to employ machine learning techniques for this specific classification task. The findings pave the way for further investigations, offering a novel approach to accelerate and optimize antibody development processes.

# Description
* train.py: training of ProtBERT and AntiBERTy model using MLP
* train_OCSVM.py: training of ProtBERT and AntiBERTy model using OCSVM
* pred.py: inference on ProtBERT and AntiBERTy
* pred_ensamble: inference using the ensamble models on ProtBERT and AntiBERty
* src/baseline_dataset.py: dataloader for csv
* src/pdb.py: pdb reader to extract the sequences from the pdb file
* src/protbert.py: class for ProtBERT and AntiBERTy
