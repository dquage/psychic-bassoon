package dl.cnn_sentence;

import com.google.common.collect.Lists;
import dl.categorisation.LibelleWord2Vec;
import dl.categorisation.word2vec.MyWord2Vec;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.*;

public class CnnSentenceClassificationExample {


    private static LibelleWord2Vec modelWords;

    public CnnSentenceClassificationExample() throws IOException, InterruptedException {
        MyWord2Vec myWord2Vec = new MyWord2Vec();
        modelWords = new LibelleWord2Vec(myWord2Vec);
    }

    public void train() throws IOException, InterruptedException {

        //Basic configuration
        int batchSize = 32;
        int vectorSize = 300;               //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;                    //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345); //For shuffling repeatability

        //Load word vectors and get the DataSetIterators for training and testing
        System.out.println("Creating DataSetIterators");
        DataSetIterator trainIter = getDataSetIterator("depenses2017.data");
        DataSetIterator testIter = getDataSetIterator("depenses2017.test");

        int nbOut = trainIter.getLabels().size();

        //Set up the network configuration. Note that we have multiple convolution layers, each wih filter
        //widths of 3, 4 and 5 as per Kim (2014) paper.

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(Updater.ADAM)
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .l2(0.0001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5,vectorSize)
                        .stride(1,vectorSize)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                //MergeVertex performs depth concatenation on activations: 3x[minibatch,100,length,300] to 1x[minibatch,300,length,300]
                .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
                //Global pooling: pool over x/y locations (dimensions 2 and 3): Activations [minibatch,300,length,300] to [minibatch, 300]
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(nbOut)    // Calculé lors de la création de l'iterator
                        .build(), "globalPool")
                .setOutputs("out")
                //Input has shape [minibatch, channels=1, length=1 to 256, 300]
                .setInputTypes(InputType.convolutional(truncateReviewsToLength, vectorSize, 1))
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

//        System.out.println("Number of parameters by layer:");
//        for (Layer l : net.getLayers()) {
//            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
//        }

        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(100));
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = net.evaluate(testIter);

            System.out.println(evaluation.stats());
        }


    }

    private DataSetIterator getDataSetIterator(String filename) throws IOException, InterruptedException {

        List<String> libelles = Lists.newArrayList();
        List<String> categoriesForLibelles = Lists.newArrayList();
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "NumCompte", "Categorie")
//                .addColumnCategorical("Categorie", Arrays.asList(Main.classes))
                .build();

        //Print out the schema:
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Id", "NumCba", "Montant", "NumCompte")
//                .categoricalToInteger("Categorie")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(filename).getFile()));

        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            libelles.add(next.get(0).toString());
            categoriesForLibelles.add(next.get(1).toString());
        }

        LabeledSentenceProvider p = new CollectionLabeledSentenceProvider(libelles, categoriesForLibelles);

        return new CnnSentenceDataSetIterator.Builder()
                .sentenceProvider(p)
                .wordVectors(modelWords.getLoadedVec())
                .minibatchSize(32)
                .maxSentenceLength(modelWords.getVecLength())
                .useNormalizedWordVectors(false)
                .build();
    }

}
