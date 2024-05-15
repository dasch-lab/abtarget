# Antibody target classification from sequence information

Monoclonal antibodies are highly specific therapeutic options for treating various diseases. However, the traditional approach to developing neutralizing antibodies has limitations, including a lack of control over target selection. Improving the screening strategy is crucial, and identifying the target molecule's molecular nature can prioritize potential candidates for in-vitro characterization. In this work, we explore the use of deep learning algorithms for identifying the target nature of monoclonal antibodies based on sequence information.

# Description
* train_bootstrap.py: training and evaluation of ProtBERT and AntiBERTy models using MLP with the opportunity to perform also the bootstrap analysis (for the single training set args.bootstrap to False)
* train_OCSVM.py: training and evaluation of ProtBERT and AntiBERTy models using OCSVM
* pred_combination.py: ensemble of ProtBERT and AntiBERty for all the experiments except SMOTE augmentation
* pred_SMOTE.py: ensemble of ProtBERT and AntiBERty for SMOTE augmentation
* src/baseline_dataset.py: class SAbDabDataset to access data from a CSV file
* src/data_loading_split.py: contains the function to split the dataset (training, validation and test)
* src/metrics.py: definition of the class for the Matthew's correlation coefficient (MCC)
* src/pdb.py: parsing of the pdb to optain the sequence
* src/protbert.py: model definition
* src/training_eval.py: function to perform training and evaluation
* test_data: the data used in test are in sabdab_200423_test_norep.csv while for the SMOTE augmentation the model uses directly the embeddings (AntiBERTy: sabdab_200423_test_norep_antiberty_embeddings.csv; ProtBERT: sabdab_200423_test_norep_protbert_embeddings.csv)
