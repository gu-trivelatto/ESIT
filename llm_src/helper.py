import os
import math
import pickle
import sqlite3

import numpy as np
import pandas as pd

from abc import ABC
from CESM.core.data_access import DAO

class HelperFunctions(ABC):
    def __init__(self):
        self.debug_log_path = 'metadata/debug.log'

    def get_params_and_cs_list(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")
        
        # Put the column names (parameters) and the first row (descriptions) together in an array
        parameters = df.columns[10:-2].to_list()
        descriptions = df.iloc[0,10:-2].to_list()
        param_n_desc = []
        for i in range(len(parameters)):
            param_n_desc.append(f'{parameters[i]} - {descriptions[i]}')
            
        # Filter the conversion processes and generate a new column with the full name 'cp@cin@cout'
        df = df.loc[(df['conversion_process_name'] != 'DEBUG') & (df['conversion_process_name'].notnull())]
        df = df.iloc[:,0:3]
        df['cs'] = df[['conversion_process_name', 'commodity_in', 'commodity_out']].agg('@'.join, axis=1)
        
        return param_n_desc, df['cs'].to_list()

    def get_populated_params_and_cs_list(self, techmap_file, cs_selection):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")

        new_row = [col if type(df.iloc[0][col]) != str else f"{col} - {df.iloc[0][col]}" for col in df.columns]
        df.columns = new_row

        df = df.loc[(df['conversion_process_name'] != 'DEBUG') & (df['conversion_process_name'].notnull())]
        df['cs'] = df[['conversion_process_name', 'commodity_in', 'commodity_out']].agg('@'.join, axis=1)

        populated_params = {cs: None for cs in cs_selection}
        for i in range(len(cs_selection)):
            df_filtered = df.loc[df['cs'] == cs_selection[i]]
            populated_params[cs_selection[i]] = df_filtered.iloc[:, 10:-3].dropna(axis=1).columns.to_list()

        return populated_params

    def get_values(self, techmap_file, selection_dict):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")
        units = df.iloc[1,:]
        df = df.loc[(df['conversion_process_name'] != 'DEBUG') & (df['conversion_process_name'].notnull())]
        df['cs'] = df[['conversion_process_name', 'commodity_in', 'commodity_out']].agg('@'.join, axis=1)
        result = {}
        
        for key, value in selection_dict.items():
            df_filtered = df.loc[df['cs'] == key]
            result[key] = []
            for param in value:
                if '-' in param:
                    param_split = param.split('-')
                    param = param_split[0].strip()
                result[key].append([param, df_filtered[param].values[0], units[param]])

        return result

    def get_scenario_params(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"Scenario")

        base_index = df.index[df['scenario_name'] == 'Base'].tolist()[0]
        discount_rate = df['discount_rate'][base_index]
        annual_co2_limit = df['annual_co2_limit'][base_index]
        co2_price = df['co2_price'][base_index]
        
        if math.isnan(discount_rate):
            discount_rate = None
        if math.isnan(co2_price):
            co2_price = None

        return {'discount_rate': discount_rate, 'annual_co2_limit': annual_co2_limit, 'co2_price': co2_price}

    def get_conversion_processes(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"Commodity")

        cond = df['commodity_name'] != 'Dummy'
        cond = cond & (df['commodity_name'] != 'DEBUG')
        cond = cond & (df['commodity_name'].str.contains('Help') == False)

        return df['commodity_name'][cond].tolist()
    
    def get_yearly_variations(self, techmap_file):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")

        results = []
        for col in df.columns[10:-2]:
            cond = (~df[col].isna()) & (df[col].str.contains(';'))
            values = df[col][cond].tolist()
            CPs = df['conversion_process_name'][cond].tolist()
            cin = df['commodity_in'][cond].tolist()
            cout = df['commodity_out'][cond].tolist()
            for i in range(len(values)):
                results = results + [[f'{CPs[i]}@{cin[i]}@{cout[i]}', values[i]]]
            
        return results

    def get_cs_param_selection(self, techmap_file, cs_list, param_list):
        tmap = pd.ExcelFile(techmap_file)
        df = pd.read_excel(tmap,"ConversionSubProcess")
        result = []
        
        for cs in cs_list:
            split_cs = cs.split('@')

            if len(split_cs) > 1:
                cond = df['conversion_process_name'] == split_cs[0]
                cond = cond & (df['commodity_in'] == split_cs[1])
                cond = cond & (df['commodity_out'] == split_cs[2])
            else:
                cond = df['conversion_process_name'] == split_cs[0]
            
            for param in param_list:
                if '-' in param:
                    param_split = param.split('-')
                    param = param_split[0].strip()
                try:
                    if len(split_cs) > 1:
                        result = result + [cs, param, df[param][cond].values[0], df[param][1]]
                    else:
                        for i in range(sum(cond)):
                            idx = df[param][cond].index[i]
                            cs = f'{split_cs[0]}@{df["commodity_in"][idx]}@{df["commodity_out"][idx]}'
                            result = result + [cs, param, df[param][cond].values[i], df[param][1]]
                except:
                    pass

        return result

    def consult_info(self, query, techmap_file):
        consult_type = query['consult_type']
        
        if consult_type == 'yearly_variation':
            info = self.get_yearly_variations(techmap_file)
        elif consult_type == 'cs_param_selection':
            info = self.get_cs_param_selection(techmap_file, query['cs'], query['param'])
        else:
            info = 'Consult type not recognized'
        
        return info

    def modify_scenario_sheet(self, workbook, new_values):
        scen_sheet = workbook['Scenario']
        new_values = new_values['new_values']
        coords = ['','','','']
        
        # Find the coordinates in the spreadsheet
        for idx, row in enumerate(scen_sheet.rows):
            # Get the horizontal coordinate of each parameter
            if idx == 0:
                for i in range(len(row)):
                    if row[i].value == 'scenario_name':
                        coords[0] = row[i].coordinate[0]
                    if row[i].value == 'discount_rate':
                        coords[1] = row[i].coordinate[0]
                    if row[i].value == 'annual_co2_limit':
                        coords[2] = row[i].coordinate[0]
                    if row[i].value == 'co2_price':
                        coords[3] = row[i].coordinate[0]
            # Get the vertical coordinate of the base scenario and
            # complete the others with it
            else:
                if scen_sheet[f'{coords[0]}{idx+1}'].value == 'Base':
                    for i in range(len(coords)):
                        coords[i] = f'{coords[i]}{idx+1}'
                    break
        
        # Apply the changes to the table
        result = []
        col_names = ['discount_rate', 'annual_co2_limit', 'co2_price']
        for i in range(1,4):
            old_value = scen_sheet[coords[i]].value
            if old_value != new_values[i-1]:
                scen_sheet[coords[i]].value = new_values[i-1]
                result = result + [f'{col_names[i-1]} modified from {old_value} to {new_values[i-1]}']
        
        return workbook, result
    
    def modify_cs_sheet(self, workbook, new_params):
        cs_sheet = workbook['ConversionSubProcess']
        new_params = new_params['values']

        result = []
        # Iterate over a dict of CSs, that contain a list of lists
        # {cs_name: [[param_name, new_value, unit], [param_name, new_value, unit]]}
        for cs_name, value in new_params.items():
            for param in value:
                param_name = param[0]
                new_value = param[1]
                unit = param[2]
                
                # Skip param if the value is empty
                if not new_value:
                    continue
                
                # Treat the param name if it came with the description from the agent
                if '-' in param_name:
                    split_param = param_name.split('-')
                    param_name = split_param[0].strip()
                
                # Initialize indexes as 0
                param_idx = '0'
                cs_idx = '0'
                # Find the right row and column for the combination of cs and param
                for idx, row in enumerate(cs_sheet.rows):
                    if idx == 0:
                        for i in range(len(row)):
                            if row[i].value == param_name:
                                param_idx = row[i].coordinate
                    else:
                        if f'{row[0].value}@{row[1].value}@{row[2].value}' == cs_name:
                            cs_idx = row[0].coordinate
                
                # If any of the indexes are 0, it failed
                if param_idx == '0' or cs_idx == '0':
                    result = result + [f'{param_name} of {cs_name} not found.']
                # Else, apply the new value
                else:
                    old_value = cs_sheet[f'{param_idx[0]}{cs_idx[1:]}'].value
                    cs_sheet[f'{param_idx[0]}{cs_idx[1:]}'].value = new_value
                    
                    if new_value != old_value:
                        result = result + [f'{param_name} of {cs_name} modified from {old_value} to {new_value} ({unit})']
                
        return workbook, result
    
    def load_results(self, runs_dir_path, sim_name, variables):
    # Instantiate access to the result DBs (base and new)
        sim_name = f'{sim_name}-Base'
        db_path = os.path.join(runs_dir_path, sim_name, 'db.sqlite')
        conn = sqlite3.connect(db_path)
        dao = DAO(conn)

        # Populate the result dataframes from the DB
        for idx, variable in enumerate(variables):
            df_value = dao.get_as_dataframe(variable)
            if idx == 0:
                df = df_value.rename(columns={'value': variable})
            else:
                df[variable] = df_value['value']
        
        return df

    def fill_empty_rows(self, df_base, df_new):
        # Creates a column for aggregated cs names
        df_base['cs'] = df_base[['cp','cin','cout']].agg('@'.join, axis=1)
        df_new['cs'] = df_new[['cp','cin','cout']].agg('@'.join, axis=1)

        # Get the list of CSs of each dataframe keeping the original order
        base_cs, index = np.unique(df_base['cs'], return_index=True)
        base_cs = list(base_cs[index.argsort()])
        new_cs, index = np.unique(df_new['cs'], return_index=True)
        new_cs = list(new_cs[index.argsort()])

        # Creates the output dataframes from the original ones
        full_df_base = df_base.copy()
        full_df_new = df_new.copy()

        base_cs_y = {}
        new_cs_y = {}

        # Gets the available years for each CS of each dataframe
        for cs in base_cs:
            base_cs_y[cs] = df_base['Year'].loc[df_base['cs'] == cs].tolist()
        for cs in new_cs:
            new_cs_y[cs] = df_new['Year'].loc[df_new['cs'] == cs].tolist()

        # Get the differences between the dataframes
        diff_base_new = [item for item in base_cs if item not in new_cs]
        diff_new_base = [item for item in new_cs if item not in base_cs]

        # Iterates over the differences between the new and the old results
        # to add the missing entries
        for diff in reversed(diff_new_base):
            previous_cs = df_new['cs'].iloc[(df_new['cs'] == diff).idxmax()-1]
            cp, cin, cout = diff.split('@')

            # Adds each available year for the missing CS
            for year in reversed(new_cs_y[diff]):
                full_df_base.loc[(df_base.loc[::-1,'cs'] == previous_cs).idxmax()+0.5] = cp, cin, cout, year, 0, 0, 0, 0, 0, 0, diff
                full_df_base = full_df_base.sort_index().reset_index(drop=True)

        # Iterates over the differences between the old and the new results
        # to add the missing entries
        for diff in reversed(diff_base_new):
            previous_cs = df_base['cs'].iloc[(df_base['cs'] == diff).idxmax()-1]
            cp, cin, cout = diff.split('@')

            # Adds each available year for the missing CS
            for year in reversed(base_cs_y[diff]):
                full_df_new.loc[(df_new.loc[::-1,'cs'] == previous_cs).idxmax()+0.5] = cp, cin, cout, year, 0, 0, 0, 0, 0, 0, diff
                full_df_new = full_df_new.sort_index().reset_index(drop=True)

        # Starts iteration over the merged results to fill the missing years of everything
        for i in range(len(full_df_base)-2, 0, -1):
            cs = full_df_base.loc[i,'cs']
            year = full_df_base.loc[i,'Year']
            cp, cin, cout = cs.split('@')
            # Border case where we have a new CS but the previous didn't go until 2015
            if (cs != full_df_base.loc[i+1,'cs']) and (full_df_base.loc[i+1,'Year'] != 2015):
                while full_df_base.loc[i+1,'Year'] != 2015:
                    new_year = full_df_base.loc[i+1,'Year']-5
                    full_df_base.loc[i + 0.5] = cp, cin, cout, new_year, 0, 0, 0, 0, 0, 0, full_df_base.loc[i+1,'cs']
                    full_df_base = full_df_base.sort_index().reset_index(drop=True)
            # Border case where we have a new CS but the current doesn't go until 2060
            if (cs != full_df_base.loc[i+1,'cs']) and (year != 2060):
                counter = 0
                while full_df_base.loc[i+1,'Year'] != year+5:
                    new_year = 2060 - counter*5
                    full_df_base.loc[i + 0.5] = cp, cin, cout, new_year, 0, 0, 0, 0, 0, 0, cs
                    full_df_base = full_df_base.sort_index().reset_index(drop=True)
                    counter += 1
            # Inner case where the current CS skips years, this will fill the inner gaps
            if (cs == full_df_base.loc[i+1,'cs']) and (year != full_df_base.loc[i+1,'Year']-5):
                while year != full_df_base.loc[i+1,'Year']-5:
                    new_year = full_df_base.loc[i+1,'Year']-5
                    full_df_base.loc[i + 0.5] = cp, cin, cout, new_year, 0, 0, 0, 0, 0, 0, cs
                    full_df_base = full_df_base.sort_index().reset_index(drop=True)

        # Starts iteration over the merged results to fill the missing years of everything
        for i in range(len(full_df_new)-2, 0, -1):
            cs = full_df_new.loc[i,'cs']
            year = full_df_new.loc[i,'Year']
            cp, cin, cout = cs.split('@')
            # Border case where we have a new CS but the previous didn't go until 2015
            if (cs != full_df_new.loc[i+1,'cs']) and (full_df_new.loc[i+1,'Year'] != 2015):
                while full_df_new.loc[i+1,'Year'] != 2015:
                    new_year = full_df_new.loc[i+1,'Year']-5
                    full_df_new.loc[i + 0.5] = cp, cin, cout, new_year, 0, 0, 0, 0, 0, 0, full_df_new.loc[i+1,'cs']
                    full_df_new = full_df_new.sort_index().reset_index(drop=True)
            # Border case where we have a new CS but the current doesn't go until 2060
            if (cs != full_df_new.loc[i+1,'cs']) and (year != 2060):
                counter = 0
                while full_df_new.loc[i+1,'Year'] != year+5:
                    new_year = 2060 - counter*5
                    full_df_new.loc[i + 0.5] = cp, cin, cout, new_year, 0, 0, 0, 0, 0, 0, cs
                    full_df_new = full_df_new.sort_index().reset_index(drop=True)
                    counter += 1
            # Inner case where the current CS skips years, this will fill the inner gaps
            if (cs == full_df_new.loc[i+1,'cs']) and (year != full_df_new.loc[i+1,'Year']-5):
                while year != full_df_new.loc[i+1,'Year']-5:
                    new_year = full_df_new.loc[i+1,'Year']-5
                    full_df_new.loc[i + 0.5] = cp, cin, cout, new_year, 0, 0, 0, 0, 0, 0, cs
                    full_df_new = full_df_new.sort_index().reset_index(drop=True)
        
        # Returns the filled dataframes
        return full_df_base, full_df_new

    def get_models_variation(self, df_base, df_new, variables):
        years = np.unique(df_base['Year'])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            variations = ((df_new.iloc[:,4:-1] / df_base.iloc[:,4:-1] - 1) * 100)
        zero_to_val_idx = (df_new.iloc[:,4:-1] != 0) & (df_base.iloc[:,4:-1] == 0)
        val_to_zero_idx = (df_new.iloc[:,4:-1] == 0) & (df_base.iloc[:,4:-1] != 0)
        zero_to_zero_idx = (df_new.iloc[:,4:-1] == 0) & (df_base.iloc[:,4:-1] == 0)
        variations = variations.apply(np.int64)

        variations[zero_to_val_idx] = np.inf
        variations[val_to_zero_idx] = np.nan
        variations[zero_to_zero_idx] = 0

        df_variations = df_new.copy()
        df_variations[variables] = variations[variables]
        df_variations['cs'] = df_variations[['cp','cin','cout']].agg('@'.join, axis=1)
        
        variations_dict = {var: {year: [] for year in years} for var in variables}

        for variable in variables:
            for year in years:
                for idx, row in df_variations.loc[(df_variations['Year']==year) & (df_variations[variable]!=0)].iterrows():
                    if row['cs'] == 'DEBUG@Dummy@DEBUG':
                        continue
                    if row[variable] == np.nan:
                        entry = f'{row["cs"]} = -100%'
                    elif row[variable] == np.inf:
                        entry = f'{row["cs"]} = from 0 to {df_new.loc[idx, [variable]].values[0]:.2f}'
                    else:
                        entry = f'{row["cs"]} = {min(row[variable], 500):.2f}%'
                    variations_dict[variable][year].append(entry)
        
        return variations_dict

    def get_yearly_variations_from_results(self, df, variables):
        df_variations = df.copy()
        df_variations['Year'] = df_variations['Year'].astype(str)
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(len(df)-1, 0, -1):
                if df.loc[i, 'cs'] == df.loc[i-1, 'cs']:
                    year_after = df.loc[i, 'Year']
                    year_before = df.loc[i-1, 'Year']
                    df_variations.iloc[i:i+1,4:-1] = (df.iloc[i:i+1,4:-1].values / df.iloc[i-1:i,4:-1].values - 1) * 100
                    df_variations.loc[i, 'Year'] = f'{year_before} to {year_after}'
                else:
                    continue
        
        variations_dict = {var: {} for var in variables}

        for variable in variables:
            for idx, row in df_variations.loc[(df_variations[variable].notna())].iterrows():
                if (row['cs'] == 'DEBUG@Dummy@DEBUG') or (len(row['Year']) == 4):
                    continue
                if row[variable] == np.inf:
                    entry = f'{row["Year"]} = from 0 to {df.loc[idx, [variable]].values[0]:.2f}'
                else:
                    entry = f'{row["Year"]} = {min(row[variable], 500):.2f}%'
                if row['cs'] in variations_dict[variable].keys():
                    variations_dict[variable][row['cs']].append(entry)
                else:
                    variations_dict[variable][row['cs']] = [entry]
                    
        return variations_dict
    
    # TODO fix memory to provide only the past 5 interactions to the user
    
    def save_history(self, history):
        with open("metadata/chat_history.pkl", "wb") as f:
            pickle.dump(history, f)
    
    def save_debug(self, debug_string):
        print(debug_string)
        with open(self.debug_log_path, 'a') as f:
            f.write(f'{str(debug_string)}\n')
            
    def get_debug_log_path(self):
        return self.debug_log_path
    
    def save_chat_status(self, status):
        with open('metadata/status.log', 'w') as f:
            f.write(status)
            
    def save_simulation_status(self, status):
        with open('metadata/simulation_status.log', 'w') as f:
            f.write(status)
            
    def get_simulation_status(self):
        with open('metadata/simulation_status.log', 'r') as f:
            status = f.read()
        return status