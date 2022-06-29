# script meant to cleanup the excel files gathered from MSE and convert them to usable CSV data for machine learning

import pandas as pd
import os
import sys

komercialna = ['Комерцијална банка Скопје','Komercijalna banka Skopje']
alkaloid = ['Алкалоид Скопје','Alkaloid Skopje']
fersped =['Фершпед Скопје','Fer{ped Skopje']
granit=['Гранит Скопје','Granit Skopje']
makosped=['Макошпед Скопје','Mako{ped Skopje']
makpetrol=['Макпетрол Скопје','Makpetrol Skopje']
makedonijaturist=['Македонијатурист Скопје','Makedonija Turist Skopje']
zkpelagonia=['ЗК Пелагонија Битола','ZK Pelagonija Bitola','Zemjod. komb. Pelagonija Bitola']
ttkbanka=['ТТК Банка АД Скопје','TTK Banka AD Skopje']


companies = {'komercialna':komercialna, 'alkaloid':alkaloid,'fersped':fersped,'granit':granit,'makosped':makosped,'makpetrol':makpetrol,'makedonijaturist':makedonijaturist,'zkpelagonia':zkpelagonia,'ttkbanka':ttkbanka}

def transform_and_cleanup(walk_dir, exclude):
 # transform and cleanup
 for root, subdirs, files in os.walk(walk_dir):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    print('--\nroot = ' + root)

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

def finder(df, row):
    for col in df:
        df =  df.loc[df[col]==row[col] | (df[col].isnull() & pd.isnull(row[col]))]
        return df

def to_usable_csv():
 for root, subdirs, files in os.walk(walk_dir):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    print('--\nroot = ' + root)
    for filename in files:
        if filename.endswith((".xls", ".xlsx")):
            file_path = os.path.join(root, filename)
            to_filename = filename.replace(".xls",".csv")
            to_filename = to_filename.replace(".xlsx",".csv")
            to_file_path = os.path.join(root, to_filename)
            print('\t- file %s (full path: %s)' % (filename, file_path))
            data_xls = pd.read_excel(file_path, dtype=str, sheet_name=0, header=None, skiprows=2, usecols=[0, 2, 3,4,5], names=['name', 'max', 'min', 'start', 'close'])
            df = data_xls.dropna()
            print(df)
            #df_alkaloid = df[df['A'].isin(alkaloid)]
            submap=data_xls.loc[data_xls['name'].isin(alkaloid),['max', 'min', 'start', 'close']]
            print(submap)
                    
            #data_xls.to_csv(to_file_path, encoding='utf-8', index=False)

#start

if len(sys.argv)<2:
    sys.argv.append('.')
walk_dir = sys.argv[1]

print('walk_dir = ' + walk_dir)
print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
exclude = set(['.git','.venv'])

#transform_and_cleanup(walk_dir, exclude)
to_usable_csv()
