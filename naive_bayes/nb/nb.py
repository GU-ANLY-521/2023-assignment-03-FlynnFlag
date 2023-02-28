
import numpy as np
import pandas as pd
from pathlib import Path

class NaiveBayes:    
    def train_nb(self, df, alpha=0.1):

        self.df = df
        vocabulary = {word: idx for idx, word in
                    enumerate(set(" ".join(df['text'].tolist()).split()))}
        n_docs = df.shape[0]
        n_classes = df.shape[1]
        priors = np.array([sum(df['author'] == auth) / n_docs for auth in range(
            n_classes)])
        # Create a matrix containing all 0s called training_matrix of size (n_docs,
        # len(vocabulary)), then fill it with the counts of each word for each
        # document
        # this is the bag-of-words matrix for all the documents
        training_matrix = np.zeros(shape=(n_docs, len(vocabulary)))
        for idx, document in enumerate(df['text'].tolist()):
            for token in document.split():
                j = vocabulary[token]
                training_matrix[idx, j] += 1
        # get word counts for both classes
        word_counts_per_class = {auth: np.sum(training_matrix[np.where(
            df['author'] == auth)]) for auth in range(n_classes)}
        likelihoods = np.zeros(shape=(n_classes, len(vocabulary)))

        for token, idx in vocabulary.items():
            for auth in range(n_classes):
                count_token_idx_in_class_auth = np.sum(np.squeeze(training_matrix[
                                            np.where(df['author'] == auth), idx]))
                likelihoods[auth, idx] = (alpha +
                    count_token_idx_in_class_auth) / \
                    (alpha * (len(vocabulary) + 1) + word_counts_per_class[auth])
        return vocabulary, priors, likelihoods



    def test(self, df, vocabulary, priors, likelihoods):

        self.df=df
        class_predictions = []
        for text in df['text']:
            test_vector = np.zeros(shape=(len(vocabulary)))
            for token in text.split():
                # skip the words that do not appear in the training corpus
                if token in vocabulary:
                    idx = vocabulary[token]
                    test_vector[idx] += 1
            # compute predictions p(y|test)
            preds = test_vector.dot(np.log(likelihoods).T) + np.log(priors)
            yhat = np.argmax(preds)
            class_predictions.append(yhat)
        return class_predictions
    
    def build_dataframe(self,folder):

        self.folder=folder
        path = Path(folder)
        df_train = pd.DataFrame(columns=['author'])
        df_test = pd.DataFrame(columns=['author'])
        author_to_id_map = {'kennedy': 0, 'johnson': 1}

        def make_df_from_dir(dir_name, df):

            for f in path.glob(f'./{dir_name}/*.txt'):
                with open(f, encoding='utf-8') as fp:
                    text = fp.read()
                    if dir_name in ('kennedy', 'johnson'):
                        temp_df = pd.DataFrame({'author': dir_name, 'text': [text]})
                        df = pd.concat([df, temp_df], ignore_index=True)
                    else:
                        temp_df = pd.DataFrame({'author': str(f).split('_')[-1][
                                                        :-4],
                                                        'text': [text]})
                        df = pd.concat([df, temp_df], ignore_index=True)
            return df
        for p in path.iterdir():
            if p.name in ('kennedy', 'johnson'):
                df_train = make_df_from_dir(p.name, df_train)
            elif p.name == 'unlabeled':
                df_test = make_df_from_dir(p.name, df_test)
        # replace the strings for the author names with numeric codes (0, 1)
        df_train['author'] = df_train['author'].apply(lambda x:
                                                    author_to_id_map.get(x))
        # do the same for the test data
        df_test['author'] = df_test['author'].apply(lambda x:
                                                    author_to_id_map.get(x))
        return df_train, df_test

