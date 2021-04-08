import os
import json
import glob
import tqdm
from typing import List, Dict
from collections import defaultdict

from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import TfidfRetriever
from haystack.reader.farm import FARMReader
from haystack import Finder, finder


class QuestionAnsweringHindi:

    def __init__(self, textFilesDir: str, trainingFile: str):
        """

        :param textFilesDir: Path of dir which contain text (.txt) files (docs which supposed to be index)
        :param trainingData: Path of training json file (for fine-tuning purpose)
        """
        self.textFilesDir = textFilesDir
        self.trainingFile = trainingFile

    def get_data_haystack_format(self) -> List:
        """

        :return: Return a list of Haystack documents (specific format: refer this)
        """
        finalData = list()
        allFiles = glob.glob(os.path.join(self.textFilesDir, '*.txt'))
        for file in tqdm.tqdm(allFiles):
            tempDict = defaultdict()
            with open(file, 'r') as f:
                content = f.read()
            tempDict['text'] = content
            tempDict['meta'] = None
            finalData.append(dict(tempDict))
        return finalData

    def get_haystack_document_store(self, data, similarity_metric='cosine', do_index=True):
        """

        :param similarity_metric: Which metric to use to measure closeness either 'dot product' or 'cosine'
        :param do_index: Whether you want to index data after initializing document_store or not
        :param data: Haystack format data (list of dicts in predefined format)
        """
        # documentStore: Database to use to store & index the data i.e. ElasticSearch, InMemory, Faiss etc.
        documentStore = InMemoryDocumentStore(similarity_metric)
        if do_index:
            try:
                documentStore.write_documents(data)
                return documentStore
            except Exception as e:
                print(e)
        else:
            return documentStore

    def fine_tune_qa_model(self, outputModelName, epoch,
                           baseModel='sentence-transformers/msmarco-distilroberta-base-v2', use_gpu=True):
        """

        :param outputModelName: Fine-tuned model outputName provided by user
        :param epoch: no of epoch to fine-tune model
        :param baseModel: Base model on which you'd fine-tuning with custom data
        :param use_gpu: whether to use GPU or not (recommended to use)
        """
        reader = FARMReader(model_name_or_path=baseModel, use_gpu=-1)
        try:
            reader.train(data_dir=self.textFilesDir, train_filename=self.trainingFile, use_gpu=use_gpu, n_epochs=epoch,
                         save_dir=outputModelName)
            print('fine-tunning done!')
        except Exception as e:
            print(e)

    def get_haystack_retriever(self, document_store):
        """

        :param document_store: provide document_store where you indexed your data i.e. o/p of get_haystack_document_store()
        :return: return Haystack retriever object accordingly i.e BM25, tf-idf, DPR etc.
        """
        return TfidfRetriever(document_store=document_store)

    def get_haystack_reader(self, fine_tuned_model_path):
        """

        :param fine_tuned_model_path: location of model which you've just fine_tuned or any other you want to use i.e. sentence-transformers
        :return: return Haystack reader object.
        """
        return FARMReader(model_name_or_path=fine_tuned_model_path)

    def get_haystack_finder(self, reader, retriever):
        """

        :param reader: Haystack reader obj
        :param retriever: Haystack retriever
        :return: Haystack finder object
        """
        return Finder(reader, retriever)

    def get_answers(self, question, document_store, finder, n_retriever_result=10, n_reader_result=3):
        """

        :param question: User query
        :param finder: Haystack finder object
        :param n_retriever_result: No of results to extracted from retriever
        :param n_reader_result: No of results to be filtered with reader from retriever results
        :return: Python list of dict containing multiple properties of extracted answers.
        """
        results = finder.get_answers(question, top_k_retriever=n_retriever_result, top_k_reader=n_reader_result)
        return [{'answer': result['answer'], 'context': result['context'], 'startLoc': result['offset_start_in_doc'], 'endLoc': result['offset_end_in_doc'], 'docText':  self.get_haystack_doc_text_by_id(document_store, result['document_id']), 'probability': result['probability']} for
                result in results['answers']]

    def get_haystack_doc_text_by_id(self, document_store, docId):
        """

        :param document_store: Haystack Document Store Object
        :param docId: doc_id of Document
        :return: return the textual content of document.
        """
        return document_store.get_document_by_id(docId).to_dict()['text']


if __name__ == '__main__':
    pass
