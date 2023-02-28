import argparse
import sys
sys.path.append("..")
from nb.nb import NaiveBayes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to make corpus "
                                                 "for Naive Bayes homework")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()
    # instantiation
    nb=NaiveBayes()
    # generate data 
    training_df, test_df = nb.build_dataframe(args.indir)
    # calculate metircs
    vocabulary, priors, likelihoods = nb.train_nb(training_df)
    #make prediction
    class_predictions = nb.test(test_df, vocabulary, priors, likelihoods)
    print(class_predictions)
