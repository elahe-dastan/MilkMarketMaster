import pandas as pd

def equalisation():
    smp_quotations = pd.read_csv('../data/smp_quotations.csv')

    # Convert values from 'lb' to 'mt' where the unit is 'lb'
    smp_quotations.loc[smp_quotations['raw_unit'] == 'lb', 'raw_price'] = smp_quotations.loc[smp_quotations[
                                                                                                 'raw_unit'] == 'lb', 'raw_price'] * 2204.62  # Conversion factor: 1 lb = 0.000453592 mt

    # Convert values from 'kg' to 'mt' where the unit is 'kg'
    smp_quotations.loc[smp_quotations['raw_unit'] == '100kg', 'raw_price'] = smp_quotations.loc[smp_quotations[
                                                                                                    'raw_unit'] == '100kg', 'raw_price'] * 10  # Conversion factor: 1 kg = 0.001 mt

    smp_quotations = smp_quotations[
        ['data_series_id', 'product_id', 'data_source_id', 'date', 'raw_currency', 'raw_price', 'currency', 'price',
         'country_id']]

    smp_quotations
    print()

def IQR(df):
    Q1 = df[['raw_price', 'price']].quantile(0.25)
    Q3 = df[['raw_price', 'price']].quantile(0.75)
    IQR = Q3 - Q1

    outlier_mask = ((df[['raw_price', 'price']] < (Q1 - 1.5 * IQR)) |
                    (df[['raw_price', 'price']] > (Q3 + 1.5 * IQR)))

    filtered_df = df[~outlier_mask.any(axis=1)]

    return filtered_df[['date', 'data_interval', 'raw_currency', 'raw_unit', 'raw_price', 'currency', 'unit', 'price']]

def clean_smp_quotations():
    smp_2_4 = pd.read_csv('../data/2/4/smp_quotations.csv')
    IQR(smp_2_4)



