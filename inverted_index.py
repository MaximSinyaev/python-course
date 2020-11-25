#!/usr/bin/env python3
import os
import re
import json
import sys
import struct
from collections import defaultdict
from typing import List
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType

LETTERS_RE = re.compile(r'[\w\s^]')
SPACES_SQUEEZE_RE = re.compile(r'\s+')
OUTPUT = sys.stderr
KEY_LEN_SIZE = 1
KEY_CHAR_SIZE = 1
DOC_ID_LEN_SIZE = 2
DOC_ID_SIZE = 2


class StoragePolicy:
    """
    Parent cls for data saving, provides template for further using
    """

    @staticmethod
    def dump(word_to_docs_mapping, filepath: str):
        """
        Dumps inverted index table

        :param word_to_docs_mapping: inverted index table
        :param filepath: path for file to save
        :return: None
        """

    @staticmethod
    def load(filepath: str):
        """
        Load inverted index table from filepath

        :param filepath: path to saved inverted index
        :return:
        """


class BinaryStoragePolicy(StoragePolicy):
    """
    Child class of StoragePolicy, that uses JSON format as output and input
    """

    @staticmethod
    def dump(word_to_docs_mapping, filepath: str):
        """
        Dumps inverted index table

        :param word_to_docs_mapping: inverted index table
        :param filepath: path for file to save
        :return: None
        """
        result = b''.join([
            struct.pack(f'B{len(key.encode("utf8"))}s', len(key.encode('utf8')), key.encode('utf8')) + \
            struct.pack(f'H{len(word_to_docs_mapping[key])}H', len(word_to_docs_mapping[key]),
                        *word_to_docs_mapping[key])
            for key in word_to_docs_mapping])
        # result = b''
        # for key in word_to_docs_mapping:
        #     key_len = len(key.encode('utf8'))
        #     word_mapping_bytes = struct.pack(f'B{key_len}s', key_len, key.encode('utf8'))
        #     doc_len = len(word_to_docs_mapping[key])
        #     doc_id_bytes = struct.pack(f'H{doc_len}H', len(word_to_docs_mapping[key]), *word_to_docs_mapping[key])
        #     result += (word_mapping_bytes + doc_id_bytes)
        #     # print(f"Key: {key_len}, {key.encode('utf8')}; Docs: {doc_len}, {word_to_docs_mapping[key]}")
        with open(filepath, 'wb') as fout:
            fout.write(result)

    @staticmethod
    def load(filepath: str):
        """
        Load inverted index table from filepath

        :param filepath: path to saved inverted index
        :return:
        """
        index_table = defaultdict(set)
        with open(filepath, 'rb') as fin:
            encoded = fin.read()
            i = 0
            while i < len(encoded):
                key_len = struct.unpack("B", encoded[i:i + 1])[0]
                i += 1
                key = struct.unpack(f'{key_len}s', encoded[i: i + (key_len * KEY_CHAR_SIZE)])[0].decode('utf8')
                i += (key_len * KEY_CHAR_SIZE)
                doc_len = struct.unpack("<H", encoded[i:i + 1 * DOC_ID_LEN_SIZE])[0]
                i += 1 * DOC_ID_LEN_SIZE
                index_table[key] = [id_ for id_ in struct.unpack(f'<{doc_len}H', encoded[i: i + doc_len * DOC_ID_SIZE])]
                # print(f"{key_len}, {key}, docs: {doc_len}, {list(index_table[key])[:10]}")
                i += doc_len * DOC_ID_SIZE
        return index_table


class JsonStoragePolicy(StoragePolicy):
    """
    Child class of StoragePolicy, that uses JSON format as output and input
    """

    @staticmethod
    def dump(word_to_docs_mapping, filepath: str):
        """
        Dumps inverted index table

        :param word_to_docs_mapping: inverted index table
        :param filepath: path for file to save
        :return: None
        """
        with open(filepath, 'w') as fout:
            json.dump(word_to_docs_mapping, fout)

    @staticmethod
    def load(filepath: str):
        """
        Load inverted index table from filepath

        :param filepath: path to saved inverted index
        :return:
        """
        with open(filepath, 'r') as fout:
            index_table = json.load(fout)
        return index_table


class InvertedIndex:
    """
    Class for implementing inverted index
    """
    storage_policy_dict = {
        'json': JsonStoragePolicy,
        'binary': BinaryStoragePolicy
    }

    def __init__(self, index_table: dict):
        self.index_table = index_table

    def query(self, words: List):
        """Return the list of relevant documents for the given query"""
        res = set()
        for word in words:
            if self.index_table.get(word, None) is None:
                continue
            if res:
                res &= set(self.index_table[word])
            else:
                res = set(self.index_table[word])
        return res if res else None

    def dump(self, filepath: str, storage_policy='json'):
        self.check_storage_policy(storage_policy)
        storage_policy_class = self.storage_policy_dict[storage_policy]
        storage_policy_class.dump(self.index_table, filepath)

    @classmethod
    def load(cls, filepath: str, storage_policy='json'):
        check_filepath_existance(filepath)
        cls.check_storage_policy(storage_policy)
        storage = cls.storage_policy_dict[storage_policy]()
        inv_table = storage.load(filepath)
        return cls(inv_table)

    @classmethod
    def check_storage_policy(cls, storage_policy: str):
        assert storage_policy in cls.storage_policy_dict.keys(), \
            f'Storage policy must be one from list: {cls.storage_policy_dict.keys()}'

    def __eq__(self, other):
        return self.index_table == other.index_table

    def __repr__(self):
        return str(dict(self.index_table))


def check_filepath_existance(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} is not exists")


def load_documents(filepath: str):
    check_filepath_existance(filepath)
    result = dict()
    with open(filepath, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line and line != '\n':
                data = line.split('\t')
                index, text = data[0], " ".join(data[1:])
                result[int(index)] = text
    return result


def build_inverted_index(documents: dict, stopwords: List = None):
    stopwords = stopwords or []
    invert_table = defaultdict(set)
    for doc_id in documents.keys():
        line = documents[doc_id].strip()
        line = re.sub(SPACES_SQUEEZE_RE, ' ', line)
        # line = re.sub(spaces_strip, ' ', re.sub(letters_re, '', line.lower()))
        _ = [invert_table[word].add(doc_id) for word in line.split() if (word and word not in stopwords)]
    for word in invert_table.keys():
        invert_table[word] = list(invert_table[word])
    return InvertedIndex(invert_table)


def load_stopwords(filepath):
    check_filepath_existance(filepath)
    with open(filepath, 'r') as fin:
        stopwords = [word.strip() for word in fin.read().splitlines()]
    return stopwords


def setup_parser(parser: ArgumentParser):
    """
    Parser setup for CLI

    :param parser: ArgumentParser
    :return: None
    """
    subparsers = parser.add_subparsers(help='Choose command')
    # Build parser
    build_parser = subparsers.add_parser(
        "build", help="build inverted index and save in binary format on hard drive",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    build_parser.add_argument(
        '-d', '--dataset',
        dest='dataset_path',
        required=True,
        help='Load target dataset',
    )
    build_parser.add_argument(
        '--stop-words',
        dest='stop_words',
        required=False,
        help='Path for file with stop words',
    )
    build_parser.add_argument(
        '-o', '--output',
        dest='output_path',
        required=True,
        help='Path to output index',
    )
    build_parser.set_defaults(command="build", callback=build_and_dump_index_callback)
    # Query parser
    query_parser = subparsers.add_parser(
        "query", help="Query words and find in documents",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    query_parser.add_argument(
        '-i', '--index',
        dest='index_path',
        required=True,
        help='Path for ready index'
    )
    query_command_group = query_parser.add_mutually_exclusive_group(required=True)
    query_command_group.add_argument(
        "--query-file-utf8",
        nargs=1,
        dest='query_file',
        metavar="QUERY_FILE_PATH",
        type=FileType('r', encoding="utf8")
    )
    query_command_group.add_argument(
        "--query-file-cp1251",
        nargs=1,
        dest='query_file',
        metavar="QUERY_FILE_PATH",
        type=FileType('r', encoding="cp1251")
    )
    query_command_group.add_argument(
        '-q', '--query',
        nargs="+",
        metavar='WORD',
        dest='words',
        help='Search query'
    )
    query_parser.set_defaults(command="query", callback=query_callback)


def build_and_dump_index_callback(arguments):
    docs = load_documents(arguments.dataset_path)
    stopwords = load_stopwords(arguments.stop_words) if arguments.stop_words else None
    inverted_index = build_inverted_index(docs, stopwords=stopwords)
    inverted_index.dump(arguments.output_path, storage_policy="binary")
    print(arguments, file=OUTPUT)


def query_callback(arguments):
    inverted_index = InvertedIndex.load(arguments.index_path, storage_policy="binary")
    response = list()
    if arguments.words:
        response.append(inverted_index.query(arguments.words))
    else:
        for query in arguments.query_file:
            response.append(inverted_index.query(query.split()))
    _ = [print(*r) if r else print() for r in response]
    return response


def main():
    parser = ArgumentParser(
        prog="inverted_index",
        description="Util for invert index creation and word search in documents",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
