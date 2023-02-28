import sys
sys.path.append("..")
from nb.nb import NaiveBayes
import numpy as np
import pandas as pd

# load the file
def tes_nb_class():
    with open('tests/data/docs.txt', 'r') as f:
        content = f.readlines()
    label=[]
    text=[]
    # transform the text to dataframe
    for i in content:
        i=i.replace("\n","")
        text.append(i.split("): ")[1])
        label.append(int(i.split("): ")[0][-1])-1)
    df=pd.DataFrame({"author":label,"text":text})
    df

    nb=NaiveBayes()
    # use our function
    vocabulary, priors, likelihoods=nb.train_nb(df)
    priors

    #test1
    assert np.allclose(priors, [0.75, 0.25], atol=1e-8)

    # test2
    expected = np.zeros(shape=(2, len(vocabulary)))
    expected[0][vocabulary['Japan']]=0
    expected[0][vocabulary['Chinese']]=5/8
    expected[0][vocabulary['Tokyo']]=0
    expected[0][vocabulary['Shanghai']]=1/8
    expected[0][vocabulary['Macao']]=1/8
    expected[0][vocabulary['Beijing']]=1/8
    expected[1][vocabulary['Chinese']]=1/3
    expected[1][vocabulary['Shanghai']]=0
    expected[1][vocabulary['Beijing']]=0
    expected[1][vocabulary['Macao']]=0
    expected[1][vocabulary['Tokyo']]=1/3
    expected[1][vocabulary['Japan']]=1/3
    assert np.allclose(likelihoods, expected, atol=0.05)

    # test3
    assert np.allclose(np.sum(likelihoods, axis=1), [1.0, 1.0], atol=0.05)

tes_nb_class()
        



