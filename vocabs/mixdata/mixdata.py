# coding=utf8
import os
import re
import glob
import random
import argparse
from tqdm import tqdm

def clean_text_line(line: str) -> str:
    '''普通文本（如 enwik8）的简单清洗'''
    line = line.strip()
    return line if line else None


def preprocess_sql(text: str) -> str:
    '''SQL 文本清洗：去注释，保证关键符号分隔'''
    text = re.sub(r'--.*', '', text)  # 单行注释
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)  # 块注释
    sql_regex = re.compile(r'(<>|!=|<=|>=|==|=|,|;|\.|\(|\)|\{|\}|\[|\]|->>|->|::|\+|-|\*|/)')
    text = sql_regex.sub(r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_fasta(text: str) -> str:
    '''GENE 序列清洗：去掉header，保留ACGTN'''
    lines = [l.strip().upper() for l in text.splitlines() if not l.startswith('>')]
    seq = ''.join(lines)
    seq = re.sub(r'[^ACGTN]', '', seq)
    return seq if len(seq) > 0 else None


def read_sql_dir(sql_dir):
    if sql_dir is not None and os.path.exists(sql_dir):
        sql_files = glob.glob(os.path.join(sql_dir, '**/*.sql'), recursive=True)
        sql_samples = []
        for fp in tqdm(sql_files, desc='Loading SQL'):
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()
                sql_samples.append(preprocess_sql(data))
        return sql_samples
    else:
        return []


def read_fasta_dir(fasta_dir):
    if fasta_dir is not None and os.path.exists(fasta_dir):
        fasta_files = glob.glob(os.path.join(fasta_dir, '*'), recursive=True)
        gene_samples = []
        for fp in tqdm(fasta_files, desc='Loading GENE'):
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read()
                seq = preprocess_fasta(data)
                if seq:
                    gene_samples.append(seq)
        return gene_samples
    else:
        return []


def read_text_file(text_file):
    if text_file is None and os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        lines = [clean_text_line(l) for l in lines if clean_text_line(l)]
        return lines
    else:
        return []


def build_mixed_corpus(args):
    all_samples = []

    if args.data_type in ['text', 'all']:
        print(f'Reading text file: {args.text_file}')
        all_samples.extend(read_text_file(args.text_file))

    if args.data_type in ['sql', 'all']:
        print(f'Reading SQL dir: {args.sql_dir}')
        all_samples.extend(read_sql_dir(args.sql_dir))

    if args.data_type in ['gene', 'all']:
        print(f'Reading gene dir: {args.gene_dir}')
        all_samples.extend(read_fasta_dir(args.gene_dir))
        
    print(f'Collected {len(all_samples)} total samples before shuffle.')
    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(sample.strip() + '\n')

    print(f'Mixed corpus saved to: {args.output_file}')
    print(f'Total lines: {len(all_samples)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build Mixed Tokenizer Corpus')
    parser.add_argument('--text_file', type=str, default='/media/ps/ssd6/zhaoy/datasets/text/enwik8')
    parser.add_argument('--sql_dir', type=str,  default='/media/ps/ssd6/zhaoy/datasets/database/spider/train')
    parser.add_argument('--gene_dir', type=str,  default='/media/ps/ssd6/zhaoy/datasets/Gene/train')
    parser.add_argument('--output_file', type=str, default='./train_corpus.txt', help='Output merged corpus path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_type', type=str, default='all',
                        choices=['text', 'sql', 'gene', 'all'],
                        help='Which data type to process (text/sql/gene/all)')
    args = parser.parse_args()

    random.seed(args.seed)
    build_mixed_corpus(args)
