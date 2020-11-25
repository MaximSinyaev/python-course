from textwrap import dedent
from collections import namedtuple

import pytest

from inverted_index import load_documents, InvertedIndex, build_inverted_index
from inverted_index import load_stopwords, check_filepath_existance, query_callback
from inverted_index import JsonStoragePolicy, StoragePolicy

# Consts
TEST_INDEX_TABLE = {
    'bow': {1, 4, 7},
    'cbow': {2, 5, 8},
    'tfidf': {0, 2, 4, 6, 8}
}
TINY_SAMPLE_WORD_DICT = {
    'BOW': [14],
    'Bag': [14],
    'of': [1000, 14],
    'words': [1000, 14],
    'CBOW': [1000],
    'Continius': [1000],
    'bag': [1000]}
TINY_SAMPLE_INV_TABLE = InvertedIndex(TINY_SAMPLE_WORD_DICT)
WIKIPEDIA_FILEPATH = 'wikipedia_sample.txt'
SMALL_SAMPLE_FILEPATH = 'sample_1.txt'
TINY_SAMPLE_FILEPATH = 'tiny_sample.txt'
STOPWORDS_EXAMPLE = 'stopwords.txt'
NOT_REAL_FILENAME = 'dfjgjsadkgfuage13212hkjdashh.asd'


def test_read_docs_sample_v1():
    loaded_docs = load_documents(SMALL_SAMPLE_FILEPATH)
    res = {
        1: 'Article 1   Some text to test inverted index',
        2: 'Article 2   Another paragraph with no common words with first one',
        17: 'Article 3   Sample text similar to first article for test',
        5: "АФЫФЫё фывфапфва фывтлавы фывтлфы ΔG ‡"
    }
    assert res == loaded_docs


def test_load_stopwords():
    stopwords = load_stopwords(STOPWORDS_EXAMPLE)
    etalon_stopwords = ['to', 'of']
    assert etalon_stopwords == stopwords


def test_not_existing_file():
    with pytest.raises(FileNotFoundError):
        check_filepath_existance(NOT_REAL_FILENAME)
    check_filepath_existance(SMALL_SAMPLE_FILEPATH)
    with pytest.raises(FileNotFoundError):
        load_documents(NOT_REAL_FILENAME)


def test_read_docs_sample_v2(tmpdir):
    dataset_str = dedent("""\
    14	BOW Bag of words
    1000	CBOW Continius bag of words
    """)
    dataset_fio = tmpdir.join("light.dataset")
    dataset_fio.write(dataset_str)
    docs = load_documents(dataset_fio)
    etalon_docs = {
        14: "BOW Bag of words",
        1000: "CBOW Continius bag of words"
    }
    assert docs == etalon_docs


def test_query_single_word(word='bow'):
    inv_idx = InvertedIndex(TEST_INDEX_TABLE)
    doc_ids = inv_idx.query([word])
    right_answer = TEST_INDEX_TABLE[word]
    assert doc_ids == right_answer


def test_query_2_intersect_words(words=['bow', 'tfidf']):
    inv_idx = InvertedIndex(TEST_INDEX_TABLE)
    doc_ids = inv_idx.query(words)
    right_answer = {4, }
    assert doc_ids == right_answer


def test_query_2_words_without_shared_docs(words=['bow', 'cbow']):
    inv_idx = InvertedIndex(TEST_INDEX_TABLE)
    doc_ids = inv_idx.query(words)
    right_answer = None
    assert doc_ids is right_answer


def test_unseen_word(word='fasttext'):
    inv_idx = InvertedIndex(TEST_INDEX_TABLE)
    doc_ids = inv_idx.query([word])
    right_answer = None
    assert doc_ids is right_answer


def test_index_creation():
    docs = load_documents(TINY_SAMPLE_FILEPATH)
    inv_idx = build_inverted_index(docs)
    assert TINY_SAMPLE_INV_TABLE == inv_idx
    assert repr(TINY_SAMPLE_WORD_DICT) == repr(inv_idx)


def test_check_storage_policy():
    with pytest.raises(AssertionError):
        InvertedIndex.check_storage_policy('not_real_flag')
    assert InvertedIndex.check_storage_policy('json') is None


def test_dump_and_load_index(tmp_path, tiny_sample_document):
    dir = tmp_path / "tiny_example_dir"
    dir.mkdir()
    index_file = dir / "tiny_example.index"
    docs = tiny_sample_document
    inv_table = build_inverted_index(docs)
    inv_table.dump(index_file)
    assert inv_table == TINY_SAMPLE_INV_TABLE
    loaded_inv_table = InvertedIndex.load(index_file)
    assert inv_table == loaded_inv_table


def test_binary_dump_and_load_index(tmp_path, tiny_sample_document, words=['of', 'words']):
    dir_ = tmp_path / "tiny_example_dir"
    dir_.mkdir()
    index_file = dir_ / "tiny_example.bin.index"
    docs = tiny_sample_document
    inv_table = build_inverted_index(docs)
    inv_table.dump(index_file, storage_policy='binary')
    assert inv_table == TINY_SAMPLE_INV_TABLE
    loaded_inv_table = InvertedIndex.load(index_file, storage_policy='binary')
    assert inv_table == loaded_inv_table
    # test query callback
    Args = namedtuple('Args', ['index_path', 'words'])
    args = Args(index_path=index_file, words=words)
    response = query_callback(args)
    ethalon_response = [{14, 1000},]
    assert response == ethalon_response




def test_for_coverage():
    StoragePolicy.dump(None, 'somepath')
    StoragePolicy.load('somepath')


def test_invert_table_eq(tiny_sample_document):
    docs = tiny_sample_document
    inv_table1 = build_inverted_index(docs)
    inv_table2 = build_inverted_index(docs)
    assert inv_table1 == inv_table2


@pytest.fixture
def tiny_sample_document():
    tiny_documents = load_documents(TINY_SAMPLE_FILEPATH)
    return tiny_documents
