# Checks kaggle submission for formatting correctness. 
# Usage: python check.py <competiton_name, eg. mnist> <submission csv file>

import argparse
import pandas as pd

competitions = ['mnist', 'spam', 'cifar10']
config = {
    'mnist': 
            {
             'rows': 10000,
             'range': (0, 9),
             },
    'spam': 
            {'rows': 5857,
             'range': (0, 1),
             },
    'cifar10': 
            {'rows': 10000,
             'range': (0, 9),
             }
}


parser = argparse.ArgumentParser()
parser.add_argument("comp_name",
                        choices=competitions, 
                        help="Name of competition you want to check against")
parser.add_argument("fname", 
                        help="Filename")
args = parser.parse_args()


print("Competition chosen: {}".format(args.comp_name))
print("File chosen: {}".format(args.fname))

# Load data
# print("Loading csv")
df = pd.read_csv(args.fname)


# Check the column names
print("Checking column names...")
if ['Id', 'Category'] != list(df.columns.values):
    print("Column header name check failed; most likely a spelling mistake OR" + \
            " a space was put between 'Id,' and 'Category'")
    exit(1)

# Check the row count
# print("Checking row count...")
rows_count = df.shape[0]
if rows_count != config[args.comp_name]['rows']:
    print("Row count should be {} when it is {}".format(config[args.comp_name]['rows'], rows_count))
    exit(1)

# Check the indexes
if list(range(1, rows_count+1)) != list(df['Id']):
    print("Range of Ids is wrong. Most likely submission is 0-indexed instead of 1-indexed")
    exit(1)

# Check the range of the category and the Ids. 
low, hi = config[args.comp_name]['range']
if not df['Category'].between(low, hi).all(): 
    print("Range of Category is wrong")
    exit(1)

# Check the dtype
if df['Id'].dtype != 'int64':
    print("Id is not of type int64, instead is of type {}".format(df['Id'].dtype))
    exit(1)

if df['Category'].dtype != 'int64':
    print("Category is not of type int64")
    exit(1)

print("All basic sanity checks passed!")




