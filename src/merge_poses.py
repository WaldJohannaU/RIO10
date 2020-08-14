import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='./', help="input path (from the sequences)")
parser.add_argument('--output_file', type=str, default='merged_results.txt', help="result file")
parser.add_argument('--suffix', type=str, default='.txt', help="suffix of pose prediction file")
opt = parser.parse_args()

def merge():
    results_dict = {}
    output_file = open(opt.output_file, "w")
    for root, dirs, files in os.walk(opt.input_folder):
        for file in files:
            if not file.startswith(".") and file.endswith(opt.suffix):
                print("reading file", file)
                file_read = open(os.path.join(root, file), "r") 
                lines = file_read.readlines()
                for line in lines:
                    output_file.write(line)
                file_read.close()
    output_file.close()

if __name__ == "__main__": merge()