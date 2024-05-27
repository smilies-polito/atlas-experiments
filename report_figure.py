import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from seaborn.objects import Jitter
from typing import Union, Literal, Optional

def jitter(values:np.array, mean:float=0 , scale:float=1):
    """

    Function that performs jitter to values for scatterplot. Applies random normal deviations to points.

    Parameters
    ----------
    values: numpy.array 
        values to scatter
    mean: float
        mean for normal distribution
    scale: float
        standard deviation for normal distribution

    Output
    -------
    numpy array with jittered values

    """
    return values + np.random.normal(mean, scale ,values.shape)


class Risultati:

    def scatter_n_macrostates_quality(self, path: str, title: Literal['scRNA', 'scATAC', 'Multiomics+scVELO', 'Multiomics+Multivelo'] ):
        """

        Function that given a csv file containing macrostate quality results plots crispness vs minchi scatterplot for all combinations of chi, 

        Parameters
        -----------
        path: str
            path of the .csv file
        title: str
            string indicating the model 

        """
        dataframe = pd.read_csv(path, header=0)
        fig, ax = plt.subplots()
        sns.scatterplot(data = dataframe, x='minChi', y='crispness', hue='n_states', palette="tab20", ax=ax)
        plt.axhline(y=0.7, color='grey', linestyle='--')
        plt.axhline(y=0.6, color='grey', linestyle='--')
        plt.axvline(x=0.0, color= 'grey', linestyle='dotted')
        plt.title(f"{title} number of macrostates quality")
        plt.savefig(f"{title}_macrostate_quality.png")


    def read_probabilities_csv(self, path:str, lsi:float, pc:float, model:Literal['scRNA', 'scATAC', 'multiomics', 'multivelo'], celltypes: Optional[pd.DataFrame]=None):
        """

        Function that reads cell fate probabilties towards terminal states and merges with cell types annotations if provided

        Parameters:
        -----------
        path: str
            path of csv file containing cell fate probabilities.
        lsi: float
            float indicating the number of LSI dimensions
        pc: float
            float indicating the number of principal components
        model: string
            string indicating the model 
        celltypes: optional, pandas.DataFrame
            dataframe containing cell barcodes as index and cell tpyes in the columns (default is None).
        
        Output:
        -------
        tuple (bool, pandas.Dataframe) bool indicated whether the path exists, while pandas.DayaFrame is None if False.
        
        
        """
        if os.path.exists(path):
             df = pd.read_csv(path, header=0)
             df = df.rename(columns={'Unnamed: 0':'barcode'})
             df = pd.merge(df, celltypes, left_on="barcode", right_index=True, how="left") if celltypes is not None else df
             df['lsi'] = np.ones(len(df))*lsi
             df['pc'] = np.ones(len(df))*pc
             df['model'] = [model]*len(df)
             return True, df
        else:
             return False, None
        

    def to_categorical(self, dataframe:pd.DataFrame, columns=list):
        """
        
        Function that turns columns into categorical columns.

        Parameters
        ----------
        dataframe: pandas.DataFrame
        columns: list 
            list of columns to turn into pandas.Categorical

        Output
        -------
        pandas.DataFrame
        
        """
        for col in columns:
            if col in dataframe.columns:
                dataframe[col] = pd.Categorical(dataframe[col])
        return dataframe
    

    def subset_categories(self, dataframe:pd.DataFrame, column:str, values: list):
        """
        
        Function that subsets categorical dataframe according to values list.

        Parameters
        ----------
        dataframe: pandas.DataFrame
        column: string
            columns according to which values are subset
        values: list
            list of accepted values

        Output
        ------
        pandas.DataFrame

        """
        if column in dataframe.columns:
            dataframe = dataframe[dataframe[column].isin(values)]

        return dataframe
        
        

    def subset_columns_of_interest(self, dataframe:pd.DataFrame, columns:list, cumulative:bool=False, terminal_states: Optional[list]=None):
        """
         
         Function that subsets columns of interest from DataFrame.

         Parameters
         ----------
         dataframe: pandas.DataFrame
         columns: list 
            columns names
         cumulative: bool
            if cumulative is True compute cumulative probabilities (default is False)
         terminal_states: list, optional
            list with terminal states names to subset. used only when cumulative is True (default None).
         
            
        Output
        ------
        pandas.DataFrame

        """
        available_columns = [col for col in columns if col in dataframe.columns]

        if cumulative:
            dict={}
            for terminal in terminal_states:
                for name in dataframe.columns:
                    if name.startswith(terminal):
                        if not dict.__contains__(terminal):
                            dict[terminal] = dataframe[name].values
                        else:
                            dict[terminal]+= dataframe[name].values

            new_dataframe = pd.DataFrame(dict)
            for ac in available_columns:
                new_dataframe[ac] = dataframe[ac]
            
            new_dataframe.index = dataframe.index
            return new_dataframe
        
        else:
            return dataframe[available_columns]



risultati = Risultati()
k_list = [20,30,50,60,80]
pc_list=[10,15,20,25,30]
lsi_list = [10,15,20,25,30]
ns= 2

celltypes = pd.read_csv(os.path.join(os.getcwd(), 'cell_annotations.tsv'), sep="\t", header=0, index_col=0)
cell_type_key='celltype'
terminal_states = ['Deeper Layer', 'Upper Layer', 'RG, Astro, OPC']

rna_path = os.path.join(os.getcwd(), 'risultati', 'scRNA_fate_probs')
mo_path = os.path.join(os.getcwd(), 'risultati', 'multiomics_fate_probs')
mv_path = os.path.join(os.getcwd(), 'risultati', 'multivelo_fate_probs')


#############################################################################################################################################################################################
#                                                                        MULTILINEAGE POTENTIAL                                                                                             #
#############################################################################################################################################################################################
multilineage_potential_columns = ['barcode', 'pc', 'lsi', 'model', cell_type_key, 'entropy', 'KL']
types_for_multilineage  = ["Deeper Layer", "RG, Astro, OPC", "Upper Layer", "IPC"]

KL_results = os.path.join(os.getcwd(), 'KL_results')
entropy_results = os.path.join(os.getcwd(), 'entropy_results')

if not os.path.exists(KL_results):
    os.makedirs(KL_results)

if not os.path.exists(entropy_results):
    os.makedirs(entropy_results)

for k in k_list:
    for pc in pc_list:
            fig1,ax1 = plt.subplots(figsize=(10,8))
            fig2,ax2 = plt.subplots(figsize=(10,8))
            mo_complete = None
            mv_complete = None
            # scRNA-seq
            rna_exists, rna_df = risultati.read_probabilities_csv(path=os.path.join(rna_path, f"probabilities_{ns}states_{k}K{pc}PC.csv"), pc=pc, lsi=-1, model='scRNA', celltypes=celltypes)
            if rna_exists:
                rna_df = risultati.subset_columns_of_interest(rna_df, multilineage_potential_columns)
                rna_df = risultati.to_categorical(dataframe=rna_df, columns=['lsi', 'pc', 'model', cell_type_key])
                rna_df = risultati.subset_categories(dataframe=rna_df, column=cell_type_key, values=types_for_multilineage)
                sns.pointplot(data=rna_df, x=cell_type_key, y="KL", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax1)
                sns.pointplot(data=rna_df, x=cell_type_key, y="entropy", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax2)

            # Multiomics + scVELO & Multiomics + Multivelo
            for lsi in lsi_list:
                mo_exists, mo_df = risultati.read_probabilities_csv(path=os.path.join(mo_path, f"probabilities_{ns}states_{k}K{pc}PC{lsi}LSI.csv"), pc=pc, lsi=lsi, model='multiomics', celltypes=celltypes)
                if mo_exists:
                    mo_df = risultati.subset_columns_of_interest(mo_df, multilineage_potential_columns)
                    mo_df = risultati.to_categorical(dataframe=mo_df, columns=['lsi', 'pc', 'model', cell_type_key])
                    mo_df = risultati.subset_categories(dataframe=mo_df, column=cell_type_key, values=types_for_multilineage)
                    mo_complete = mo_df if mo_complete is None else pd.concat([mo_complete, mo_df])
                
                mv_exists, mv_df = risultati.read_probabilities_csv(path=os.path.join(mv_path, f"probabilities_{ns}states_{k}K{pc}PC{lsi}LSI.csv"), pc=pc, lsi=lsi, model='multivelo', celltypes=celltypes)
                if mv_exists:
                    mv_df = risultati.subset_columns_of_interest(mv_df, multilineage_potential_columns)
                    mv_df = risultati.to_categorical(dataframe=mv_df, columns=['lsi', 'pc', 'model', cell_type_key])
                    mv_df = risultati.subset_categories(dataframe=mv_df, column=cell_type_key, values=types_for_multilineage)
                    mv_complete = mv_df if mv_complete is None else pd.concat([mv_complete, mv_df])            
            
            if mo_complete is not None:
                sns.pointplot(data=mo_complete, x=cell_type_key, y="KL", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax1, markers=["x"]*len(set(mo_complete.lsi)), dodge=.4)
                sns.pointplot(data=mo_complete, x=cell_type_key, y="entropy", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax2, markers=["x"]*len(set(mo_complete.lsi)), dodge=.4)
            if mv_complete is not None:
                sns.pointplot(data=mv_complete, x=cell_type_key, y="KL", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax1, markers=["*"]*len(set(mv_complete.lsi)), dodge=.2)
                sns.pointplot(data=mv_complete, x=cell_type_key, y="entropy", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax2, markers=["*"]*len(set(mv_complete.lsi)), dodge=.2)

            ax1.set_title(f"Average KL and 0.95 Confidence Intervals K={k}, n_states={ns}, pc={pc}")
            ax2.set_title(f"Average Entropy and 0.95 Confidence Intervals K={k}, n_states={ns}, pc={pc}")
            
            fig1.savefig(os.path.join(KL_results, f"K{k}PC{pc}_{ns}.png"))
            fig2.savefig(os.path.join(entropy_results, f"K{k}PC{pc}_{ns}.png"))
            plt.close()

#############################################################################################################################################################################################
#                                                                        AVERAGE FATE PROBABILITIES                                                                                         #
#############################################################################################################################################################################################
fate_probs_columns = ['barcode', 'pc', 'lsi', 'model', cell_type_key]
fate_probs_results = os.path.join(os.getcwd(), 'fate_prob_results')

if not os.path.exists(fate_probs_results):
    os.makedirs(fate_probs_results)

for k in k_list:
    for pc in pc_list:
        fig1,ax1 = plt.subplots(figsize=(10,8))
        fig2,ax2 = plt.subplots(figsize=(10,8))
        fig3,ax3 = plt.subplots(figsize=(10,8))
        mo_complete = None
        mv_complete = None
        # scRNA-seq
        rna_exists, rna_df = risultati.read_probabilities_csv(path=os.path.join(rna_path, f"probabilities_{ns}states_{k}K{pc}PC.csv"), pc=pc, lsi=-1, model='scRNA', celltypes=celltypes)
        if rna_exists:
            rna_df = risultati.subset_columns_of_interest(rna_df, fate_probs_columns, cumulative=True, terminal_states=terminal_states)
            rna_df = risultati.to_categorical(dataframe=rna_df, columns=['lsi', 'pc', 'model', cell_type_key])
            sns.pointplot(data=rna_df, x=cell_type_key, y="Deeper Layer", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax1)
            sns.pointplot(data=rna_df, x=cell_type_key, y="RG, Astro, OPC", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax2)

        # Multiomics + scVELO & Multiomics + Multivelo
        for lsi in lsi_list:
            mo_exists, mo_df = risultati.read_probabilities_csv(path=os.path.join(mo_path, f"probabilities_{ns}states_{k}K{pc}PC{lsi}LSI.csv"), pc=pc, lsi=lsi, model='multiomics', celltypes=celltypes)
            if mo_exists:
                mo_df = risultati.subset_columns_of_interest(mo_df, fate_probs_columns, cumulative=True, terminal_states=terminal_states)
                mo_df = risultati.to_categorical(dataframe=mo_df, columns=['lsi', 'pc', 'model', cell_type_key])
                mo_complete = mo_df if mo_complete is None else pd.concat([mo_complete, mo_df])
            
            mv_exists, mv_df = risultati.read_probabilities_csv(path=os.path.join(mv_path, f"probabilities_{ns}states_{k}K{pc}PC{lsi}LSI.csv"), pc=pc, lsi=lsi, model='multivelo', celltypes=celltypes)
            if mv_exists:
                mv_df = risultati.subset_columns_of_interest(mv_df, fate_probs_columns, cumulative=True, terminal_states=terminal_states)
                mv_df = risultati.to_categorical(dataframe=mv_df, columns=['lsi', 'pc', 'model', cell_type_key])
                mv_complete = mv_df if mv_complete is None else pd.concat([mv_complete, mv_df])   
      

            if mo_complete is not None:
                sns.pointplot(data=mo_complete, x=cell_type_key, y="Deeper Layer", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax1, markers=["x"]*len(set(mo_complete.lsi)), dodge=.4)
                sns.pointplot(data=mo_complete, x=cell_type_key, y="RG, Astro, OPC", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax2, markers=["x"]*len(set(mo_complete.lsi)), dodge=.4)
                if "Upper Layer" in mo_complete.columns:
                    sns.pointplot(data=mo_complete, x=cell_type_key, y="Upper Layer", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax3, markers=["x"]*len(set(mo_complete.lsi)), dodge=.4)

            if mv_complete is not None:
                sns.pointplot(data=mv_complete, x=cell_type_key, y="Deeper Layer", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax1, markers=["*"]*len(set(mv_complete.lsi)), dodge=.2)
                sns.pointplot(data=mv_complete, x=cell_type_key, y="RG, Astro, OPC", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax2, markers=["*"]*len(set(mv_complete.lsi)), dodge=.2)
                if "Upper Layer" in mv_complete.columns:
                    sns.pointplot(data=mv_complete, x=cell_type_key, y="Upper Layer", hue="lsi", linestyles='none', estimator="mean", errorbar=("ci",95), ax=ax3, markers=["*"]*len(set(mv_complete.lsi)), dodge=.2)

        ax1.set_title(f"Cluster-wise average fate probability towards Deeper Layer K={k}, n_states={ns}, pc={pc}")
        ax2.set_title(f"Cluster-wise average fate probability towards RG, Astro, OPC K={k}, n_states={ns}, pc={pc}")
        ax3.set_title(f"Cluster-wise average fate probability towards Upper Layer K={k}, n_states={ns}, pc={pc}")

        fig1.savefig(os.path.join(fate_probs_results, f"DL_K{k}PC{pc}_{ns}.png"))
        fig2.savefig(os.path.join(fate_probs_results, f"RG_K{k}PC{pc}_{ns}.png"))
        fig3.savefig(os.path.join(fate_probs_results, f"UL_K{k}PC{pc}_{ns}.png"))
        plt.close()


            
            
            
                