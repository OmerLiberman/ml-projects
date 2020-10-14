import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store', dest='data', required=True)
parser.add_argument('--type', action='store', dest='type')
parser.add_argument('--test', action='store', dest='test_size', type=float, default=0.2)
args = parser.parse_args()

data_dir = args.data
all_sub_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(data_dir + '/' + d) and not d.startswith('.cnvrg')]
print("All sub directories found: " ,all_sub_dirs)

## Creating the new directories.
os.mkdir(data_dir + '/train_set')
os.mkdir(data_dir + '/test_set')
for d in all_sub_dirs:
	os.mkdir(data_dir + '/train_set/' + d)
	os.mkdir(data_dir + '/test_set/' + d)


## Moving images.
for d in all_sub_dirs:
	print("Dividing: {}".format(d))

	files_in_d = [img for img in os.listdir(data_dir + '/' + d) if img.endswith(args.type)]

	test_set_size = int(len(files_in_d) * args.test_size)

	test_set = files_in_d[:test_set_size]
	train_set = files_in_d[test_set_size:]

	## Moving the train set.
	for img in train_set:
		shutil.move(src=data_dir + '/' + d + '/' + img, dst=data_dir + '/train_set/' + d + '/' + img)

	## Moving the train set.
	for img in test_set:
		shutil.move(src=data_dir + '/' + d + '/' + img, dst=data_dir + '/test_set/' + d + '/' + img)

	print("\t\tDivided: {} successfully!".format(d))

	### Deletes the previous directories.
	shutil.rmtree(data_dir + '/' + d)
