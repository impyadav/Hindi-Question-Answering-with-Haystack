import streamlit as st
from annotated_text import annotated_text
from qa_main_class import QuestionAnsweringHindi


def get_results(inputQuery):
    """

    :param inputQuery: Question by user
    :return: return list of dict of extracted answers from document store with their probability, answer location in text etc.
    """
    res = qaObj.get_answers(inputQuery, DS, FINDER_NEW)
    return res


def filtered_results(response_content):
    returnResult = []
    for item in response_content['answers']:
        returnResult.append((item['answer'], item['docText'], item['probability']))
    return returnResult


def get_annotated_text(text, keyword, probability):
    """

    :param text: doc text
    :param keyword: corresponding answer
    :param probability: probability associated with answer respectively
    :return: st-annotated-text format (to display highlighted component on streamlit web app)
    """
    temp = text.split(keyword)
    temp.insert(1, ((keyword, probability, '#afa'), ''))
    tempList = list(tuple(temp[0].split()) + temp[1] + tuple(temp[2].split()))
    tempList.remove('')
    return tuple([item + ' ' if type(item) != tuple else item for item in tempList])


def get_highlighted_answers(list_of_tuples):
    """

    :param list_of_tuples: o/p of filtered_results method
    :return: list of annotated_text objects.
    """
    return {'answers': [annotated_text(*get_annotated_text(item[1], item[0], item[2])) for item in list_of_tuples]}


if __name__ == '__main__':
    qaObj = QuestionAnsweringHindi('path_to_textDir', 'path_to_training_json_file')
    data_to_index = qaObj.get_data_haystack_format()
    DS = qaObj.get_haystack_document_store(data_to_index)
    RETRIEVER = qaObj.get_haystack_retriever(DS)
    READER = qaObj.get_haystack_reader('path_of_your_fine_tuned_model')
    FINDER_NEW = qaObj.get_haystack_finder(READER, RETRIEVER)

    st.title('Question-Answering System')
    st.write('Language: {} || Content: {}'.format('Hindi', 'History Books'))
    st.subheader("Input")
    user_input = st.text_area('', height=25)  # height in pixel
    result = filtered_results(get_results(user_input))
    annotated_text = get_highlighted_answers(result)
    if st.button('Run'):
        st.write(result)
