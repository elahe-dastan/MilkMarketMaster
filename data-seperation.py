import os

import pandas as pd


def smp_quotations_separation():
    smp_quotations = pd.read_csv('./data/smp_quotations.csv')

    unique_combinations = smp_quotations[['country_id', 'product_id']].drop_duplicates()

    for index, row in unique_combinations.iterrows():
        country_id = row['country_id']
        product_id = row['product_id']

        subset_df = smp_quotations[
            (smp_quotations['country_id'] == country_id) & (smp_quotations['product_id'] == product_id)]

        target_file_dir = './data' + '/' + str(country_id) + '/' + str(product_id)

        os.makedirs(target_file_dir, exist_ok=True)

        subset_df.to_csv(target_file_dir + '/smp_quotations.csv')


def consumption_separation():
    consumption = pd.read_csv('./data/consumptions.csv')

    unique_combinations = consumption[['country_id', 'product_id']].drop_duplicates()

    for index, row in unique_combinations.iterrows():
        country_id = row['country_id']
        product_id = row['product_id']

        subset_df = consumption[
            (consumption['country_id'] == country_id) & (consumption['product_id'] == product_id)]

        target_file_dir = './data' + '/' + str(country_id) + '/' + str(product_id)

        os.makedirs(target_file_dir, exist_ok=True)

        subset_df.to_csv(target_file_dir + '/consumptions.csv')


def production_separation():
    production = pd.read_csv('./data/production.csv')

    unique_combinations = production[['country_id', 'product_id']].drop_duplicates()

    for index, row in unique_combinations.iterrows():
        country_id = row['country_id']
        product_id = row['product_id']

        subset_df = production[
            (production['country_id'] == country_id) & (production['product_id'] == product_id)]

        target_file_dir = './data' + '/' + str(country_id) + '/' + str(product_id)

        os.makedirs(target_file_dir, exist_ok=True)

        subset_df.to_csv(target_file_dir + '/production.csv')


def inflation_separation():
    inflation = pd.read_csv('./data/inflation_rates.csv')

    unique_countries = inflation['country_id'].unique()

    for country_id in unique_countries:
        subset_df = inflation[inflation['country_id'] == country_id]

        target_file_dir = './data' + '/' + str(country_id)

        os.makedirs(target_file_dir, exist_ok=True)

        subset_df.to_csv(target_file_dir + '/inflation_rates.csv')


def separation(data_path, file_name, *data_columns):
    smp_quotations = pd.read_csv(data_path)
    data_columns = list(data_columns)

    unique_combinations = smp_quotations[data_columns].drop_duplicates()
    # Create a new DataFrame to store the filtered data without outliers
    # filtered_df = pd.DataFrame()
    # Step 2-5: Iterate over unique combinations, calculate IQR, and drop outliers
    for index, row in unique_combinations.iterrows():
        country_id = row['country_id']
        product_id = row['product_id']

        # Step 3: Subset the DataFrame based on current combination
        subset_df = smp_quotations[
            (smp_quotations['country_id'] == country_id) & (smp_quotations['product_id'] == product_id)]
        target_file_name = './data' + '/' + str(country_id) + '/' + str(product_id) + '/' + file_name + '.csv'
        subset_df.to_csv(target_file_name, index=False)


# smp_quotations_separation()
# consumption_separation()
# production_separation()
inflation_separation()
