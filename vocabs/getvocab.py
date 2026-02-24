# coding=utf8
import os
import json
import logging
import subprocess
import sentencepiece as spm
from collections import OrderedDict

from tqdm import tqdm

class PretrainVocab:
    def __init__(
        self, 
        vocab_root, 
        vocab_dir = None,
        prefix_name = 'spm', 
        train_file = '/media/ps/ssd6/zhaoy/datasets/enwik8', 
        test_file   = '/media/ps/ssd6/zhaoy/datasets/enwik9', 
        vocab_size = 16384, 
        model_type = 'bpe', 
        coverage   = 1,
        unk_id = 1,
        byte_fallback = False,
        logger     = logging.getLogger('base')
    ):
        self.logger     = logger
        self.train_file = train_file
        self.test_file   = test_file
        
        self.vocab_root  = vocab_root
        self.vocab_dir   = vocab_dir
        self.vocab_size  = vocab_size
        self.prefix_name = prefix_name
        self.model_type  = model_type
        self.coverage    = coverage 
        self.unk_id      = unk_id  
        self.byte_fallback = byte_fallback    
        self.subdir = 'unk_deny' if self.byte_fallback else 'unk_allow' 
        
        self.vocab_dir = f'{self.vocab_root}/{self.subdir}/vocab_{self.prefix_name}_{self.model_type}_{self.vocab_size}_{self.coverage}' if vocab_dir is None else vocab_dir
        os.makedirs(self.vocab_dir, exist_ok=True) 
        
    def get_train_info(self):
        with open(self.train_file, 'r', encoding='utf-8') as f:
            lines  = f.readlines() 
        max_len = max([len(line) for line in lines])
        print(f'There are {len(lines)} lines in {self.train_file}, the length of the longest sentence is {max_len}')
    
    def do_spm_training(
        self, 
        spm_bin='/home/zhaoy/OmniZip-CVPR2026/vocabs/spm_train',
        text_only=False
    ):  
        if text_only:
            cmd = [
                    spm_bin,
                    f"--input={self.train_file}",
                    "--pad_id=0",
                    f"--unk_id={self.unk_id}",
                    "--bos_id=2",
                    "--eos_id=-1",
                    f"--model_prefix={self.vocab_dir}/{self.prefix_name}_{self.model_type}_{self.vocab_size}_{self.coverage}",
                    f"--vocab_size={self.vocab_size}",
                    f"--character_coverage={self.coverage}",
                    "--max_sentence_length=10000",
                    "--add_dummy_prefix=0",
                    "--remove_extra_whitespaces=0",
                ]
        else:
            pixel_values = [str(i) for i in range(256)]
            sql_keywords = [
                "SELECT","FROM","WHERE","JOIN","INSERT","UPDATE","DELETE",
                "CREATE","TABLE","VALUES","GROUP","BY","ORDER","AND","OR","NOT"
            ]
            sql_ops = [
                "<>", "<=", ">=", "!=", "||", "&&", "::", "->>", "->", ":=",
                "(", ")", "[", "]", "{", "}", ";", ".", "*", "%", "+", "-", "/", "=", "<", ">"
            ]
            dna_bases = ["A", "C", "G", "T", "N"]
            modal_separators = ["<image>", "<medical>", "<tactile>", "<speech>", "<database>", "<gene>", "<text>"]
            user_defined_symbols = sql_keywords + sql_ops + dna_bases + pixel_values + modal_separators
            
            cmd = [
                    spm_bin,
                    f"--input={self.train_file}",
                    "--pad_id=0",
                    f"--unk_id={self.unk_id}",
                    "--bos_id=2",
                    "--eos_id=-1",
                    f"--byte_fallback={self.byte_fallback}",
                    f"--model_prefix={self.vocab_dir}/{self.prefix_name}_{self.model_type}_{self.vocab_size}_{self.coverage}",
                    f"--vocab_size={self.vocab_size}",
                    f"--character_coverage={self.coverage}",
                    "--max_sentence_length=10000",
                    "--add_dummy_prefix=0",
                    "--remove_extra_whitespaces=0",
                    f"--model_type={self.model_type}",
                    f"--user_defined_symbols={','.join(user_defined_symbols)}"
            ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        
        
    def gen_dictory(self):
        spm_vocab_path = f'{self.vocab_dir}/{self.prefix_name}_{self.model_type}_{self.vocab_size}_{self.coverage}.vocab'
        assert os.path.exists(spm_vocab_path), f'Do not exist: {spm_vocab_path}.'
        
        self.symb2id_dict = OrderedDict()
        with open(spm_vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\t\t0\n':
                    self.symb2id_dict['\t'] = len(self.symb2id_dict)
                elif line == '\n':
                    self.symb2id_dict['\n'] = len(self.symb2id_dict)
                elif line == '\t0\n':
                    continue
                else:
                    symb = line.strip().split()[0]
                    self.symb2id_dict[symb] = len(self.symb2id_dict)
                
        with open(spm_vocab_path.replace('.vocab', '.json'), 'w') as f:
            json.dump(self.symb2id_dict, f, indent=4, ensure_ascii=False)
            
            
    def generate_tokens(
        self,
        train_name = 'spm_enwik8',
        corpus_dir = '/home/zhaoy/OmniZip-CVPR2026/corpus/unk_allow',
        test_name  = None,
        text_only  = False,
        generate_train = True,
        generate_test  = True,
        encoding = 'utf-8',
        max_len = -1
    ):
        test_name = 'spm_enwik9' if test_name is None else test_name
        sp_processor = spm.SentencePieceProcessor()
        sp_processor.Load(f'{self.vocab_dir}/{self.prefix_name}_{self.model_type}_{self.vocab_size}_{self.coverage}.model')
        
        os.makedirs(corpus_dir, exist_ok=True)
        text_only_label = 'text_' if text_only else ''
        
        # generate training data
        if generate_train:
            train_vocab_file = f'{corpus_dir}/{train_name}_{text_only_label}{self.model_type}_{self.vocab_size}_{self.coverage}.txt'
            train_unk_file   = f'{corpus_dir}/{train_name.replace("spm", "unk")}_{text_only_label}{self.model_type}_{self.vocab_size}_{self.coverage}.txt'
            
            # treat enwik8 as a line
            with open(self.train_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
                
            train_tokens = []
            unk_tokens   = []
            for line in tqdm(lines, mininterval=10.0, miniters=100):
                proto  = sp_processor.encode(line, out_type='immutable_proto')
                for piece in proto.pieces:
                    if piece.begin != piece.end:
                        token = str(piece.id)
                        if token == str(self.unk_id):
                            unk_tokens.append(line[piece.begin:piece.end])
                        else:
                            train_tokens.append(token)
                
            train_tokens = ','.join(train_tokens)
            with open(train_vocab_file, 'w', encoding=encoding) as f:
                f.write(train_tokens + '\n')
            print(f'train vocabs saved to {train_vocab_file}.')
        
            unk_tokens = ''.join(unk_tokens)
            with open(train_unk_file, 'w', encoding=encoding) as f:
                f.write(unk_tokens + '\n')
            print(f'train unknown tokens saved to {train_unk_file}.')
            
        # generate test data
        if generate_test:
            test_vocab_file = f'{corpus_dir}/{test_name}_{text_only_label}{self.model_type}_{self.vocab_size}_{self.coverage}.txt'
            test_unk_file   = f'{corpus_dir}/{test_name.replace("spm", "unk")}_{text_only_label}{self.model_type}_{self.vocab_size}_{self.coverage}.txt'
            
            # treat enwik9 as a line
            with open(self.test_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
                
            # test_lines  = lines[test_range[0] : test_range[1]]
            test_tokens = []
            unk_tokens  = []
            stop_processing = False
            for line in tqdm(lines):
                if stop_processing:
                    break
                line = line.strip('\n')
                proto  = sp_processor.encode(line, out_type='immutable_proto')
                for piece in proto.pieces:
                    if piece.begin != piece.end:
                        token = str(piece.id)
                        if token == str(self.unk_id):
                            unk_tokens.append(line[piece.begin:piece.end])
                        else:
                            test_tokens.append(str(token))
                    if max_len >= 0 and len(test_tokens) >= max_len:
                        stop_processing = True
                        break
                
            test_tokens = ','.join(test_tokens)
            with open(test_vocab_file, 'w', encoding=encoding) as f:
                f.write(test_tokens + '\n')
            print(f'test vocabs saved to {test_vocab_file}.')
            
            if len(unk_tokens) > 0:
                unk_tokens = ''.join(unk_tokens)
                with open(test_unk_file, 'w', encoding=encoding) as f:
                    f.write(unk_tokens + '\n')
                print(f'test vocabs saved to {test_unk_file}.')
            else:
                print(f'There are no unknown tokens in {test_name}.')
            
    def forward(
        self,
        train_name = 'spm_enwik8',
        test_name  = None,
        get_train_info  = True,
        do_spm_training = True,
        generate_tokens = True,
        text_only = False,
        corpus_dir = '/home/zhaoy/OmniZip-CVPR2026/corpus/unk_allow',
        encoding = 'utf-8',
        max_len  = -1
    ):
        if get_train_info:
            self.get_train_info()
        if do_spm_training:
            self.do_spm_training(text_only=text_only)
        
        generate_train = True if do_spm_training else False
        if generate_tokens:
            self.gen_dictory()
            self.generate_tokens(
                train_name=train_name, 
                test_name=test_name, 
                text_only=text_only, 
                generate_train=generate_train, 
                generate_test=True,
                corpus_dir=corpus_dir, 
                encoding=encoding,
                max_len=max_len
            )
    
                    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('tokenization')
    parser.add_argument('--vocab_size', type=int,  default=16384)
    parser.add_argument('--coverage',   type=float, default=1.0)
    
    parser.add_argument('--name', type=str, nargs=2, help='[train, test]', default=['spm_enwik8', 'spm_enwik9'])
    parser.add_argument('--corpus_root', type=str,  default='/home/zhaoy/OmniZip-CVPR2026/corpus')
    parser.add_argument('--raw_file', type=str, nargs=2, help='[train, test]', 
                        default=['/media/ps/ssd6/zhaoy/datasets/text/enwik8', '/media/ps/ssd6/zhaoy/datasets/text/enwik9'])
    parser.add_argument('--encoding', type=str, default='utf-8')
    
    parser.add_argument('--text_only',   action='store_true')
    parser.add_argument('--get_train_info',  action='store_true')
    parser.add_argument('--do_spm_training', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--byte_fallback', action='store_true')
    parser.add_argument('--max_len', type=lambda x: int(float(x)), default=-1)

    # parser.add_argument('--generate_tokens', action='store_true')
    args = parser.parse_args()
    
    vocab_size = args.vocab_size
    coverage   = args.coverage
    text_only  = args.text_only
    args.generate_tokens = True

    pp = PretrainVocab(
        prefix_name='spm',
        vocab_size=vocab_size,
        coverage=coverage,
        vocab_root='/home/zhaoy/OmniZip-CVPR2026/vocabs', 
        # vocab_dir=f'/home/zhaoy/OmniZip-CVPR2026/vocabs/vocab_spm_bpe_{vocab_size}_{coverage}',
        train_file=args.raw_file[0],
        test_file=args.raw_file[1],
        byte_fallback=args.byte_fallback
    )
    unk_type = 'unk_deny' if args.byte_fallback else 'unk_allow'
    pp.forward(
        get_train_info=args.get_train_info, 
        do_spm_training=args.do_spm_training,
        generate_tokens=args.generate_tokens, 
        train_name=args.name[0], 
        test_name=args.name[1], 
        text_only=text_only,
        corpus_dir=os.path.join(args.corpus_root, unk_type), 
        encoding=args.encoding,
        max_len=args.max_len
    )