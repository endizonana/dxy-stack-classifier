import pandas as pd

def save_backtests(
    output_excel: str,
    res_gbc: pd.DataFrame,
    res_soft: pd.DataFrame,
    df_stack: pd.DataFrame,
    avg_ens: pd.DataFrame,
    backtest_stack_logit_2y: pd.DataFrame,
    backtest_stack_logit_3y: pd.DataFrame
) -> None:
    with pd.ExcelWriter(output_excel) as writer:
        if not res_gbc.empty:  res_gbc.to_excel(writer, sheet_name='gbc_monthstarts')
        if not res_soft.empty: res_soft.to_excel(writer, sheet_name='softmax_monthstarts')
        if not df_stack.empty: df_stack.to_excel(writer, sheet_name='stack_inputs')
        if not avg_ens.empty:  avg_ens.to_excel(writer, sheet_name='avg_ensemble')
        if not backtest_stack_logit_2y.empty:
            backtest_stack_logit_2y.to_excel(writer, sheet_name='stacked_logit_2y')
        if not backtest_stack_logit_3y.empty:
            backtest_stack_logit_3y.to_excel(writer, sheet_name='stacked_logit_3y')

def save_importances(output_importance_excel: str, imp_df: pd.DataFrame) -> None:
    if imp_df is not None and not imp_df.empty:
        imp_df.to_excel(output_importance_excel, sheet_name='Permutation')
