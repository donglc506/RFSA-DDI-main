# RFSA-DDI:substructure-aware graph neural network incorporating relation features for drug-drug interaction prediction
## Note
If you want to use the code in the cold start scenario, you should add dropout to MLP layers and use weight decay. 

## Requirements  

numpy==1.24.4 \
tqdm==4.65.0 \
pandas==2.0.3 \
rdkit==2020.09.1 \
scikit_learn==1.2.2 \
python==3.8.13 \
torch==1.12.0 \
torch_geometric==2.0.4 \
torch_scatter==2.0.9

## Step-by-step running:  
### 1. DrugBank
- First, cd RFSA-DDI/drugbank, and run data_preprocessing_cold.py using  
  `python data_preprocessing_cold.py -d drugbank -o all`  
  Running data_preprocessing_cold.py convert the raw data into graph format. 
- Second, run train.py using \
  `python train.py --fold 0 --save_model` 
  
### 2. TWOSIDES
- First, cd RFSA-DDI/twosides, and run data_preprocessing.py using  
  `python data_preprocessing.py -d twosides -o all`   
  Running data_preprocessing.py convert the raw data into graph format.
- Second, run train.py using \
  `python train.py --fold 0 --save_model` 
