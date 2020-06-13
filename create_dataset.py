import csv
from shutil import copyfile

with open('covid.csv') as covidfile:
	reader = csv.reader(covidfile, delimiter='\n', quotechar='|')
	header = next(reader)[0][:-1].split(',')
	folder_col = header.index('folder')
	file_col = header.index('filename')
	print(folder_col, file_col)
	for i, row in enumerate(reader):
		row = row[0][:-1].split(',')
		src_path = './' + row[folder_col] + '/' + row[file_col]
		dst_path = './covid/' + 'img' + str(i) + '.jpg'
		copyfile(src_path, dst_path)