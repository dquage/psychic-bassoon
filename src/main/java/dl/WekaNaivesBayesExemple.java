package dl;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 * Utilisation de Weka pour classifier en utilisatn l'algo NAIVE BAYES.
 * https://www.codingame.com/playgrounds/6734/machine-learning-with-java---part-5-naive-bayes
 */
public class WekaNaivesBayesExemple {

    /** file names are defined */
    public static final String TRAINING_DATA_SET_FILENAME = "depenses2017.data";
    public static final String TESTING_DATA_SET_FILENAME = "depenses2017.test";

    /**
     * This method is to load the data set.
     *
     * @param fileName
     * @return
     * @throws Exception
     */
    public static Instances getDataSet(String fileName) throws Exception {
        /**
         * we can set the file i.e., loader.setFile("finename") to load the data
         */
        StringToWordVector filter = new StringToWordVector();
        int classIdx = 5;
        /** the arffloader to load the arff file */
        CSVLoader loader = new CSVLoader();
        /** load the traing data */
        loader.setSource(WekaNaivesBayesExemple.class.getResourceAsStream(fileName));
        /**
         * we can also set the file like loader3.setFile(new
         * File("test-confused.arff"));
         */
        //loader.setFile(new File(fileName));
        Instances dataSet = loader.getDataSet();
        /** set the index based on the data given in the arff files */
        dataSet.setClassIndex(classIdx);
        dataSet.deleteAttributeAt(0);
        dataSet.deleteAttributeAt(1);
        dataSet.deleteAttributeAt(2);
        dataSet.deleteAttributeAt(4);

        filter.setInputFormat(dataSet);
        dataSet = Filter.useFilter(dataSet, filter);
        return dataSet;
    }

    /**
     * This method is used to process the input and return the statistics.
     *
     * @throws Exception
     */
    public static void process() throws Exception {

        Instances trainingDataSet = getDataSet(TRAINING_DATA_SET_FILENAME);
        Instances testingDataSet = getDataSet(TESTING_DATA_SET_FILENAME);
        Instances predictingDataSet = getDataSet(TESTING_DATA_SET_FILENAME);
        /** Classifier here is Linear Regression */
        Classifier classifier = new NaiveBayesMultinomial();
        /** */
        classifier.buildClassifier(trainingDataSet);
        /**
         * train the alogorithm with the training data and evaluate the
         * algorithm with testing data
         */
        Evaluation eval = new Evaluation(trainingDataSet);
        eval.evaluateModel(classifier, testingDataSet);
        /** Print the algorithm summary */
        System.out.println("** Naive Bayes Evaluation with Datasets **");
        System.out.println(eval.toSummaryString());
        System.out.print(" the expression for the input data as per alogorithm is ");
        System.out.println(classifier);
        for (int i = 0; i < predictingDataSet.numInstances(); i++) {
            System.out.println(predictingDataSet.instance(i));
            double index = classifier.classifyInstance(predictingDataSet.instance(i));
            String className = trainingDataSet.attribute(0).value((int) index);
            System.out.println(className);
        }

    }
}
