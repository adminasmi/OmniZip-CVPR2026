import os
import json
import logging

from PIL import Image
from datetime import datetime
from collections import OrderedDict

def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(logger_name, dir, phase, level=logging.INFO, to_screen=False, to_file=False):
    """ set-up the logger """
    lg = logging.getLogger(logger_name)
    lg.setLevel(level)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    if lg.hasHandlers():
        lg.handlers.clear()

    if to_file:
        log_path = os.path.join(dir, f"{phase}_{get_timestamp()}.log")
        fHandler = logging.FileHandler(log_path, mode='w')
        fHandler.setFormatter(formatter)
        lg.addHandler(fHandler)
        print(f"Logger '{logger_name}' will log to file: {log_path}")

    if to_screen:
        sHandler = logging.StreamHandler()
        sHandler.setFormatter(formatter)
        lg.addHandler(sHandler)
        if not to_file:
            print(f"Logger '{logger_name}' will only log to screen")
    
    lg.info(f"Logger '{logger_name}' initialized: screen={to_screen}, file={to_file}")
    return lg


### mapping words into IDs. ###
def read_word2id_dict(dict_path):
    with open(dict_path, "r") as obj_f:
        word2id_dict = json.load(obj_f)
    return word2id_dict


def gen_word2id_dict_ascii():
    """
    Generate word->Id dict in an ASCII manner.
    
    word2id_dict = {
        b'\x00': 3,  # 索引从 3 开始，因为 '<s>'、'<unk>'、'<pad>' 已占用 0, 1, 2
        b'\x01': 4,
        b'\x02': 5,
        ...
        b'\xff': 258
    }
    """
    word2id_dict = {}
    word2id_dict['<s>'] = len(word2id_dict)
    word2id_dict['<unk>'] = len(word2id_dict)
    word2id_dict['<pad>'] = len(word2id_dict)
    
    # assign a unique id for each single Byte (range 0~255). byte_order = 'big'
    for i in range(256):
        byte_val = i.to_bytes(1, 'big')
        word2id_dict[byte_val] = len(word2id_dict)
        
    return word2id_dict


### model tools. ###
def check_keep(key_name, ignorekeywords):
    for keyword in ignorekeywords:
        if keyword in key_name:
            ignorekeywords.append(key_name)
            return False
    return True


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]   # remove `module.`
        new_state_dict[k] = v
        
    return new_state_dict


def tokenize_rgb_image(image_path):
    """
    Tokenize an image by converting each pixel's RGB values into strings.
    
    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list of tokens representing the image's pixel RGB values.
              Each token is a string representation of an R, G, or B value.
    """
    # open the image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    tokens = []
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            tokens.append(str(r))
            tokens.append(str(g))
            tokens.append(str(b))
            
    return tokens


