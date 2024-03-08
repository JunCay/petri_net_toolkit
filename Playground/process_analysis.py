import csv
import os
from collections import Counter

# def read_csv_column(filename, column_index):
#     column_data = []
#     with open(filename, 'r') as file:
#         csv_reader = csv.reader(file)
#         for row in csv_reader:
#             if row[0] == 'No':
#                 continue
#             if len(row) > column_index:
#                 column_data.append(row[column_index])
#     return column_data


# current_dir = os.getcwd()
# filename = os.path.join(current_dir, 'Playground/SemiconductorProcess.csv')
# column_index = 3
# column_data = read_csv_column(filename, column_index)

process_flow='1-2-13-14-23-15-20-22-23-22-17-13-23-16-24-23-22-17-1-8-4-22-22-1-2-8-13-14-18-23-15-16-23-18-22-1-1-13-14-23-15-16-24-23-22-17-1-2-8-9-21-22-1-4-22-22-1-2-13-14-23-15-16-24-24-23-22-17-24-1-2-7-1-3-22-13-15-23-22-22-22-17-13-14-18-23-15-16-20-23-1-17-1-1-3-13-14-16-24-23-22-17-9-21-1-3-13-14-15-23-15-16-24-23-22-17-1-3-13-14-23-15-16-23-15-16-24-23-22-17-1-3-10-22-12-6-22-6-1-1-4-10-19-23-1-10-13-14-16-21-12-13-14-18-23-15-15-15-16-19-23-22-17-11-13-14-15-21-23-5'
column_data = process_flow.split('-')

res = []
res_c = []
for i in range(2, 5):
    for j in range(len(column_data)-i):
        current_cmp = []
        count = 0
        for k in range(i):
            current_cmp.append(column_data[j+k])
        for current_check in range(j+1,len(column_data)-i):
            flag = True
            for k in range(i):
                if current_cmp[k] != column_data[current_check+k]:
                    flag = False
                    break
            if flag:
                count += 1
                
        if count > 3:
            if (current_cmp) not in res:
                res.append((current_cmp))
                res_c.append((current_cmp, count))
                print(current_cmp, count)
            
                