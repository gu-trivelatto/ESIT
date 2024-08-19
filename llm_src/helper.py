import math
import pandas as pd
from abc import ABC

class HelperFunctions(ABC):
    def __init__(self):
        pass

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
                    
                    result = result + [f'{param_name} of {cs_name} modified from {old_value} to {new_value} ({unit})']
                
        return workbook, result
    
    def save_debug(self, debug_string):
        print(debug_string)
        with open('debug.log', 'a') as f:
            f.write(f'{str(debug_string)}\n')