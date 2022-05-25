#importing pandas as pd
import pandas as pd
import os
import sys

walk_dir = sys.argv[1]

print('walk_dir = ' + walk_dir)

# If your current working directory may change during script execution, it's recommended to
# immediately convert program arguments to an absolute path. Then the variable root below will
# be an absolute path as well. Example:
# walk_dir = os.path.abspath(walk_dir)
print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
exclude = set(['.git','.venv'])

for root, subdirs, files in os.walk(walk_dir):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    print('--\nroot = ' + root)
    for filename in files:
        if filename.endswith('.csv'):
            delete_path = os.path.join(root, filename)
            os.remove(delete_path)

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
                    to_file_path = os.path.join(root, to_filename)
                    os.rename(from_file_path, to_file_path)
                if filename.endswith("xlsx"):
                    to_filename=filename[-13:]
                    from_file_path = os.path.join(root, filename)
                    to_filename_year=to_filename[:4]
                    to_filename_month=to_filename[4:6]
                    extension = filename.split(".")[-1]
                    to_file_path = os.path.join(root, to_filename)
                    os.rename(from_file_path, to_file_path)

            #file_path = os.path.join(root, filename)
            #to_filename = filename.replace(".xls",".csv")
            #to_filename = to_filename.replace(".xlsx",".csv")
            #to_file_path = os.path.join(root, to_filename)
            #print('\t- file %s (full path: %s)' % (filename, file_path))
            #data_xls = pd.read_excel(file_path,  dtype=str, index_col=None)
            #data_xls.to_csv(to_file_path, encoding='utf-8', index=False)
