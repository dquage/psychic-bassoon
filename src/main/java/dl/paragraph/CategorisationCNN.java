package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import dl.categorisation.word2vec.MyWord2Vec;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Record;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Catégorisation neuronale !
 * A partir du modèle WordVectors généré par ParagraphVectors.
 */
public class CategorisationCNN {

    public static List<String> CATEGORIES = Apollon.labels();
    private WordVectors wordVectors;

    public CategorisationCNN() {
    }

    public void evaluate() throws Exception {

        if (wordVectors == null) {
            MyWord2Vec myWord2vec = new MyWord2Vec();
            wordVectors = myWord2vec.getWord2Vec();
        }

        if (wordVectors == null) {
            throw new Exception("Unable to load Word Vectors");
        }

        if (CATEGORIES == null) {
            throw new Exception("Il faut d'abord renseigner la liste complète des catégories pour évaluer");
        }

        List<Record> testRecords  = readCSVRecords("apollon_data_2018.test.csv");
        List<Record> trainRecords = readCSVRecords("apollon_data_2018.train.csv");
        List<Record> allRecords = new ArrayList<>(testRecords);
        allRecords.addAll(trainRecords);

//        List<String> labels = getCategories(allRecords);
        List<String> labels = Arrays.asList(new String[]{"ENTRETIEN ET REPARATIONS", "FRAIS DE TELECOMMUNICATION", "COTISATION CARPIMKO", "ASSURANCES", "COMPTE DE L EXPLOITANT"});

        int batchSize = 1000;     //Number of examples in each minibatch
        int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;   //Size of the word vectors
        int vobabSize = wordVectors.vocab().numWords();
        System.out.println("Vector size : "+vectorSize);
        System.out.println("Vocab size : "+vobabSize);

        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateLibellesToLength = 30;  //Truncate reviews with length (# words) greater than this
        final int seed = 0;     //Seed for reproducibility
        int nbClasses = CATEGORIES.size();
        System.out.println("Num. classes : "+nbClasses);

        CnnSentenceDataSetIterator test = this.createCnnIterator(testRecords, labels);
        CnnSentenceDataSetIterator train = this.createCnnIterator(trainRecords, labels);

        System.out.println("Train: "+train.getLabelClassMap().size());
        System.out.println("Test: "+test.getLabelClassMap().size());

        int cnnLayerFeatureMaps = 10;
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .activation(Activation.RELU)
                .updater(new Adam(0.01, 0.9D, 0.999D, 1.0E-8D))
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
                        .poolingType(PoolingType.MAX)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nOut(labels.size())    //2 classes: positive or negative
                        .build(), "globalPool")
                .setOutputs("out")
                //Input has shape [minibatch, channels=1, length=1 to 256, 300]
                .setInputTypes(InputType.convolutional(truncateLibellesToLength, vectorSize, 1))
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("Config built");

        net.init();
        net.setListeners(new ScoreIterationListener(1));

        System.out.println("Training starting...");
        for (int i = 0; i < nEpochs; i++) {
            Stopwatch started = Stopwatch.createStarted();
            net.fit(train);
            train.reset();
            System.out.println("Training done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
            System.out.println("Epoch " + i + " complete.");
            System.out.println("Evaluation starting...");
            started.reset().start();

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = net.evaluate(test);
            System.out.println("Evaluation done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
            System.out.println(evaluation.stats());
        }

        saveModel(net);
    }

    private void saveModel(ComputationGraph net) throws IOException {
        File file = new File("model/cnn_model.zip");
        ModelSerializer.writeModel(net, file, true);
    }

    private ComputationGraph loadModel() throws IOException {
        File file = new File("model/cnn_model.zip");
        return ModelSerializer.restoreComputationGraph(file);
    }

    private CnnSentenceDataSetIterator createCnnIterator(List<Record> records, List<String> labels) {
        List <String> sentences = Lists.newArrayList();
        List <String> labelsForSentences = Lists.newArrayList();
        for (Record record : records) {
            if (labels.contains(record.getCategorie())) {
                sentences.add(record.getLibelle());
                labelsForSentences.add(record.getCategorie());
            }
        }
        CollectionLabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labelsForSentences);

        return new CnnSentenceDataSetIterator.Builder()
                .sentenceProvider(sentenceProvider)
                .wordVectors(wordVectors)
                .minibatchSize(1000)
                .maxSentenceLength(30)
                .useNormalizedWordVectors(false)
                .build();
    }

    /**
     * Retourne la liste des catégories de tous les records.
     * @param records
     * @return
     */
    private static List<String> getCategories(List<Record> records) {
        Set<String> uniqueLabels = Sets.newHashSet();
        for (Record record : records) {
            uniqueLabels.add(record.getCategorie());
        }
        return Lists.newArrayList(uniqueLabels);
    }

    /**
     * Fichier de la forme : CATEGORIE,LIBELLE,MONTANT
     *
     * @param csvFileClasspath
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public List<Record> readCSVRecords(String csvFileClasspath) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Categorie", "Libelle")
                .addColumnDouble("Montant")
                .addColumnsString("Compte", "Type")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        List<Record> records = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            records.add(Record.builder()
                    .type(Record.Type.fromString(next.get(4).toString()))
                    .categorie(next.get(0).toString())
                    .libelle(next.get(1).toString())
                    .montant(next.get(2).toDouble())
                    .numeroCompte(next.get(3).toString())
                    .build());
        }
        return records;
    }

}
