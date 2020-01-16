from unittest import TestCase

from mlprimitives.custom.text import TextCleaner


class TextCleanerTest(TestCase):

    def test_detect_language_en(self):
        texts = [
            'this is an english text',
            'this is also something written in engilsh',
            'and finally something else to make sure that the language is detected'
        ]

        language = TextCleaner.detect_language(texts)

        assert language == 'en'

    def test_detect_language_es(self):
        texts = [
            'esto es un texto en español',
            'esto también está escrito en español',
            'y finalmente alguna cosa más para asegurar que el idioma es detectado'
        ]

        language = TextCleaner.detect_language(texts)

        assert language == 'es'

    def test__remove_stopwords_empty(self):
        text_cleaner = TextCleaner()

        returned = text_cleaner._remove_stopwords('')

        assert returned == ''

    def test__remove_stopwords_not_empty(self):
        text_cleaner = TextCleaner()
        text_cleaner.language_code = 'en'

        text_cleaner.STOPWORDS['en'] = ['is', 'a', 'in']

        returned = text_cleaner._remove_stopwords('This is a text written in English')

        assert returned == 'This text written English'
