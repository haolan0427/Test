from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six

def convert_to_unicode(text):
    '''
    将text转换成unicode格式
    text(str or unicode python2/str or bytes python3):输入文本
    '''
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    '''
    将text转换成适合打印或日志记录的格式
    text(str or unicode python2/str or bytes python3):输入文本
    '''
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    '''
    从指定的词汇文件中加载词汇
    vocab_file(str):词汇文件的路径
    '''
    vocab = collections.OrderedDict()

    index_vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "rb") as reader:
        while True:
            tmp = reader.readline()
            token = convert_to_unicode(tmp) # 调用convert_to_unicode()函数

            if not token:
                break

            token = token.strip()
            vocab[token] = index # 词汇作为key；词汇索引作为value
            index_vocab[index]=token # 词汇索引作为key；词汇作为value
            index += 1


    return vocab,index_vocab # vocab词汇；index_vocab词汇索引


def convert_tokens_to_ids(vocab, tokens):
    '''
    将词汇转换成对应的索引
    vocab(dict):词汇表，词汇是key，索引是value
    tokenss(list):待转换的词汇
    '''
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids


def whitespace_tokenize(text):
    '''
    将文本按空白字符（空格、制表符）分割成单独的词汇
    text(str):输入文本
    '''
    text = text.strip()
    if not text:
        return []
    tokens = text.split() # 词汇列表，为list类型
    return tokens


class FullTokenizer(object):
    '''
    结合基本分词器（将文本按特定字符进行切分）和WordPiece分词器（将单词拆分为更小的子单元——subword）
    '''

    def __init__(self, vocab_file, do_lower_case=True):
        '''
        构造函数：
        vocab_file(str):用该路径下的文件来生成词汇表
        do_lower_case(bool):是否将文本转换为小写
        '''
        self.vocab,self.index_vocab = load_vocab(vocab_file) # 调用load_vocab()函数
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens) # 调用convert_tokens_to_ids()函数


class BasicTokenizer(object):
    '''
    实现基本分词器
    '''
    def __init__(self, do_lower_case=True):
        '''
        构造函数
        do_lower_case(bool):是否将文本转换为小写
        '''
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        '''
        对输入文本进行分词处理
        text():输入文本
        '''
        text = convert_to_unicode(text) # 调用convert_to_unicode函数
        text = self._clean_text(text) # 调用_clean_text函数
        text = self._tokenize_chinese_chars(text) # 调用_tokenize_chinese_chars函数
        orig_tokens = whitespace_tokenize(text) # 调用whitespace_tokenize函数
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token) # 调用_run_strip_accents函数
            split_tokens.extend(self._run_split_on_punc(token)) # 调用_run_split_on_punc函数

        output_tokens = whitespace_tokenize(" ".join(split_tokens)) # 调用whitespace_tokenize函数
        return output_tokens

    def _run_strip_accents(self, text):
        '''
        去除重音符号
        text(str):输入文本
        '''
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        '''
        将文本分割成单词和标点符号
        text(str):输入文本
        '''
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char): # 调用_is_punctuation函数
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]
    
    def _tokenize_chinese_chars(self, text):
        '''
        将中文字符前后添加空格，对非中文字符不进行操作
        text(str):输入文本
        '''
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp): # 调用_is_chinese_char函数
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        '''
        判读输入字符是否是中文
        cp(unicode):输入字符
        '''
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
    
        return False
    
    def _clean_text(self, text):
        '''
        将输入文本的空白字符替换为单个空格，删除控制字符
        text(str):输入文本
        '''
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    '''
    实现WordPiece分词器：能有效解决OOV问题
    '''
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        '''
        构造函数
        vocab(dict):词汇表，词汇为key，索引为value
        unk_token(str):表示未知词的标记
        max_input_chars_per_word(int):表示每个单词最大字符数
        '''
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        '''
        对输入文本进行WordPiece分词处理
        text():输入文本
        '''
        text = convert_to_unicode(text) # 调用convert_to_unicode函数
        output_tokens = []
        for token in whitespace_tokenize(text): #调用whitespace_tokenize函数
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 以下是WordPiece分词的过程：
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    '''
    判断给定字符是否为空白字符
    char(char):输入字符
    '''
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    '''
    判断给定字符是否为控制字符
    char(char):输入字符
    '''
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    '''
    判断给定字符是否为标点符号
    char(char):输入字符
    '''
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
