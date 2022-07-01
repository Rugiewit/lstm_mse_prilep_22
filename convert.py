# script meant to cleanup the excel files gathered from MSE and convert them to usable CSV data for machine learning

import pandas as pd
import os
import sys

komercialna = ['Комерцијална банка Скопје','Komercijalna banka Skopje']
alkaloid = ['Алкалоид Скопје','Alkaloid Skopje']
#fersped =['Фершпед Скопје','Fer{ped Skopje']
granit=['Гранит Скопје','Granit Skopje']
makpetrol=['Макпетрол Скопје','Makpetrol Skopje']
makedonijaturist=['Македонијатурист Скопје','Makedonija Turist Skopje','Makedonijaturist Skopje']
#replek = ['Replek Skopje','Реплек Скопје']
#skopskipazar=['Skopski Pazar Skopje','Скопски Пазар Скопје']
#zkpelagonia=['ЗК Пелагонија Битола','ZK Pelagonija Bitola','Zemjod. komb. Pelagonija Bitola']
#ttkbanka=['ТТК Банка АД Скопје','TTK Banka AD Skopje']


companies_desc = {'komercialna':komercialna, 'alkaloid':alkaloid,'granit':granit,'makpetrol':makpetrol,'makedonijaturist':makedonijaturist}

def transform_and_cleanup(walk_dir, exclude):
 # transform and cleanup
 for root, subdirs, files in os.walk(walk_dir):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    #print('--\nroot = ' + root)

    for filename in files:
        # cleanup old CSV
        if filename.endswith('.csv'):
            delete_path = os.path.join(root, filename)
            os.remove(delete_path)
        # rename excels for easier transformations
        if filename.endswith((".xls", ".xlsx")):
            if filename.startswith("ReportMK_3_m.01"):
                to_filename=filename[-11:]
                from_file_path = os.path.join(root, filename)
                to_file_path = os.path.join(root, to_filename)
                os.rename(from_file_path, to_file_path)
            if filename.startswith("ReportMK_3_20"):
                if filename.endswith("xls"):
                    to_filename=filename[-12:]
                    from_file_path = os.path.join(root, filename)
                    to_filename_year=to_filename[:4]
                    to_filename_month=to_filename[4:6]
                    extension = filename.split(".")[-1]
                    to_filename = ".".join((to_filename_month,to_filename_year,extension))
                    to_file_path = os.path.join(root, to_filename)
                    os.rename(from_file_path, to_file_path)
                if filename.endswith("xlsx"):
                    to_filename=filename[-13:]
                    from_file_path = os.path.join(root, filename)
                    to_filename_year=to_filename[:4]
                    to_filename_month=to_filename[4:6]
                    extension = filename.split(".")[-1]
                    to_filename = ".".join((to_filename_month,to_filename_year,extension))
                    to_file_path = os.path.join(root, to_filename)
                    os.rename(from_file_path, to_file_path)
            if " " in filename:
                cutNr = -8
                if filename.endswith("xlsx"):
                    cutNr = -9
                lowerCaseName=filename.lower()
                month = lowerCaseName.split(" ")[0]
                mapper = {"januari":"01","fevruari":"02","mart":"03","april":"04","maj":"05","мај":"05","juni":"06","juli":"07","avgust":"08","septemvri":"09","oktomvri":"10","noemvri":"11","dekemvri":"12"}
                to_filename_month = mapper[month]
                to_filename=filename[cutNr:]

                from_file_path = os.path.join(root, filename)
                to_filename_year=to_filename[:4]
                extension = filename.split(".")[-1]
                to_filename = ".".join((to_filename_month,to_filename_year,extension))
                to_file_path = os.path.join(root, to_filename)
                os.rename(from_file_path, to_file_path)



def trim_all_columns(df):
    """
    Trim whitespace from ends of each value across all series in dataframe
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

def to_usable_csv():
 companies = {}
 for key,value in companies_desc.items():
     companies[key]=[]    
     
 for root, subdirs, files in os.walk(walk_dir):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    #print('--\nroot = ' + root)
    for filename in files:        
        if filename.endswith((".xls", ".xlsx")):
            file_path = os.path.join(root, filename)
            to_filename = filename.replace(".xls",".csv")
            to_filename = to_filename.replace(".xlsx",".csv")
            to_file_path = os.path.join(root, to_filename)
            #print('\t- file %s (full path: %s)' % (filename, file_path))
            data_xls = pd.read_excel(file_path, dtype=str, sheet_name=0, header=None, skiprows=2, usecols=[0,  2, 3,4,5], names=['name', 'max', 'min', 'start', 'close'])
            #cleanup
            df = data_xls.dropna()
            df = trim_all_columns(df)

            #companies_names
            for c_key, c_value in companies_desc.items():
                monthdate = filename.split('.')[1]+'/'+filename.split('.')[0]
                company_row=df.loc[df['name'].isin(c_value)].explode('name').iloc[0].values.flatten().tolist()
                companies[c_key].append([monthdate]+company_row[1:5])
    
 print( companies.keys())
 for key,value in companies.items():
     to_csv_path = os.path.join(walk_dir, key)+".csv"
     print(to_csv_path)
     df =  pd.DataFrame(columns=['date',  'max', 'min', 'start', 'close'], data=value)
     df = df.sort_values('date')
     df.to_csv(to_csv_path, encoding='utf-8', index=False)

#start

if len(sys.argv)<2:
    sys.argv.append('.')
walk_dir = sys.argv[1]

print('walk_dir = ' + walk_dir)
print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
exclude = set(['.git','.venv'])

transform_and_cleanup(walk_dir, exclude)
to_usable_csv()
