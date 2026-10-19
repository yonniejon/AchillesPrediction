from data_helper import get_intersecting_gene_ids_and_data, get_tissue_types, create_train_test_df_using_manual_input
from Models import train_test_eval
import pandas as pd
import numpy as np

if __name__ == '__main__':
    essentiality_data_file = 'CRISPR_gene_effect.csv'
    expression_data_file = 'CCLE_expression.csv'
    achilles_scores, gene_expression, \
    train_test_df, cv_df = get_intersecting_gene_ids_and_data(essentiality_data_file,
                                                              expression_data_file,
                                                              cv_df_file=None,
                                                              train_test_df_file=None,
                                                              num_folds=0)

    tissue_types, tissue_count = get_tissue_types(gene_expression)
    # keep the labels out of the data frames, feature selection would treat an extra column as a gene
    tissue_types = np.array(tissue_types)
    cell_ids = np.array(gene_expression['Unnamed: 0'])

    target_gene = "RPP25L"

    sorted_tissue_count = list(sorted(tissue_count.items(), key=lambda x: -x[1]))
    top_tissues = sorted_tissue_count[:10]
    gene_rmse_dict = {"RPP25L": 0.2355486856, "VRK1": 0.3193518146, "SNAI2": 0.1602959911, "PRKAR1A": 0.2394600598}
    tissue_rmse = []
    for tissue, _ in top_tissues:
        # leave one tissue out: train on all other tissues, test on this one
        train_ids = list(cell_ids[tissue_types != tissue])
        test_ids = list(cell_ids[tissue_types == tissue])
        train_test_df = create_train_test_df_using_manual_input(train_ids, test_ids)
        cv_rmse, cv_pearson, pearson_p_val, model, features = train_test_eval(achilles_scores, gene_expression,
                                                                              target_gene,
                                                                              train_test_df,
                                                                              'xg_boost', use_knn=False,
                                                                              should_plot=False,
                                                                              num_features=15)
        tissue_rmse.append((tissue, cv_rmse))


    x = 0