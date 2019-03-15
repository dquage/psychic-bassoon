package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import dl.paragraph.my.Word2IdDataSetIterator;
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
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Catégorisation neuronale !
 * A partir du modèle WordVectors généré par ParagraphVectors.
 */
public class CategorisationW2I {

    public static List<String> CATEGORIES = Apollon.labels();
    private WordVectors wordVectors;

    public void evaluate() throws Exception {

        if (CATEGORIES == null) {
            throw new Exception("Il faut d'abord renseigner la liste complète des catégories pour évaluer");
        }

        List<Record> testRecords = readCSVRecords("apollon_data_2018.test.csv");
        List<Record> trainRecords = readCSVRecords("apollon_data_2018.train.csv");
        List<Record> allRecords = new ArrayList<>(testRecords);
        allRecords.addAll(trainRecords);

        printDataStats(trainRecords, testRecords);

        List<String> labels = getCategories(allRecords);
//        List<String> labels = Arrays.asList(new String[]{"ENTRETIEN ET REPARATIONS", "FRAIS DE TELECOMMUNICATION", "COTISATION CARPIMKO", "ASSURANCES", "COMPTE DE L EXPLOITANT"});
//        Collections.sort(labels);
        int nbClasses = labels.size();
        System.out.println("Num. classes : "+nbClasses);

        HashMap<String, Integer> word2Id = this.learnWord2Id(allRecords, 0);
        System.out.println("Vocab size : "+word2Id.size());


        int batchSize = 100;     //Number of examples in each minibatch
        int nEpochs = 1000;        //Number of epochs (full passes of training data) to train on
        int truncateLibellesToLength = 30;  //Truncate reviews with length (# words) greater than this


//        Nd4j.getMemoryManager().setAutoGcWindow(5000);  //https://deeplearning4j.org/workspaces

        Word2IdDataSetIterator train = new Word2IdDataSetIterator(trainRecords, labels, batchSize, truncateLibellesToLength, false, false, word2Id);
        Word2IdDataSetIterator test  = new Word2IdDataSetIterator(testRecords,  labels, batchSize, truncateLibellesToLength, false, false, word2Id);


//        for (int i=0; i<labels.size(); i++)
//            System.out.println(train.getLabels().indexOf(labels.get(i))+" <=> "+test.getLabels().indexOf(labels.get(i))+" <=> "+test.getLabelsClassMap().get(labels.get(i))+" ===> "+labels.get(i));
//        System.exit(0);

//        System.out.println("-----------  Concordance  ------------------");
//        String cat;
//        cat = "PART PRIVEE DES DEPENSES MIXTES"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "HONORAIRES"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "FOURNITURES DE BUREAU ET ADMINISTRATIVES"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "COTISATIONS COMPLEMENTAIRES SANTE"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "FRAIS DE FORMATION SEMINAIRE"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));



//        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAGRAD)
//                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .learningRate(0.05)
//                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(word2Id.size()).nOut(50).build())
                .layer(1, new DenseLayer.Builder().nIn(50).nOut(500).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
//                .layer(2, new DenseLayer.Builder().nIn(500).nOut(nbClasses*2).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(2, new RnnOutputLayer.Builder().nIn(500).nOut(nbClasses).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
//                .pretrain(true)
                .backprop(true)
                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
                .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
                .build();

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .activation(Activation.RELU)
//                .list()
//                .layer(0, new EmbeddingLayer.Builder().nIn(word2Id.size()).nOut(500).build())
//                .layer(1, new DenseLayer.Builder().nIn(500).nOut(500).activation(Activation.SOFTSIGN).build())
//                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(500).nOut(nbClasses).activation(Activation.SOFTMAX).build())
//                .inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
////                .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
//                .build();


//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(12345)
//                .iterations(1)
//                .weightInit(WeightInit.XAVIER)
//                .updater(Updater.ADAGRAD)
//                .activation(Activation.RELU)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .learningRate(0.05)
//                .regularization(true).l2(0.0001)
//                .list()
//                .layer(0, new EmbeddingLayer.Builder().nIn(word2Id.size()).nOut(500).activation(Activation.TANH).build())
//                .layer(1, new OutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).nIn(500).nOut(nbClasses).build())
////                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).nIn(500).nOut(nbClasses).build())
//                .build();


        System.out.println("Config built X");


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));


        System.out.println("Training starting...");
        for (int i = 0; i < nEpochs; i++) {
            Stopwatch started = Stopwatch.createStarted();

            train.reset();
            net.fit(train);

            System.out.println("Training done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
            System.out.println("Epoch " + i + " complete.");
            System.out.println("Evaluation starting...");
            started.reset().start();

            //Run evaluation. This is on 25k reviews, so can take some time
            test.reset();
            Evaluation evaluation = net.evaluate(test);
            System.out.println("Evaluation done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
            System.out.println(evaluation.stats());
        }

        saveModel(net);
    }

    private void saveModel(MultiLayerNetwork net) throws IOException {
        File file = new File("model/w2i_model.zip");
        ModelSerializer.writeModel(net, file, true);
    }

    private MultiLayerNetwork loadModel() throws IOException {
        File file = new File("model/w2i_model.zip");
        return ModelSerializer.restoreMultiLayerNetwork(file);
    }

    private HashMap<String, Integer> getCatFreq(List<Record> records) {
        HashMap<String, Integer> catFreq = Maps.newHashMap();
        for (Record record : records) {
            String cat = record.getCategorie();
            if (catFreq.containsKey(cat)) {
                catFreq.put(cat, catFreq.get(cat) + 1);
            } else {
                catFreq.put(cat, 1);
            }
        }
        return catFreq;
    }

    private void printDataStats(List<Record> train, List<Record> test) {
        HashMap<String, Integer> trainFreq = this.getCatFreq(train);
        HashMap<String, Integer> testFreq = this.getCatFreq(test);

        System.out.println("\n\n\n-----------  Train  ------------------");
        System.out.println("Num categories: "+trainFreq.size());
        System.out.println("Category frequencies:");
        for(Map.Entry<String, Integer> entry : trainFreq.entrySet()) {
            System.out.println(entry.getKey()+" : "+entry.getValue());
        }

        System.out.println("\n\n\n-----------  Test  ------------------");
        System.out.println("Num categories: "+testFreq.size());
        System.out.println("Category frequencies:");
        HashMap<String, Integer> missing = Maps.newHashMap();
        for(Map.Entry<String, Integer> entry : testFreq.entrySet()) {
            System.out.println(entry.getKey()+" : "+entry.getValue());
            if (!trainFreq.containsKey(entry.getKey())) {
                missing.put(entry.getKey(), 1);
            }
        }

        System.out.println("\n\n\n-----------  Missing  ------------------");
        System.out.println("Num categories: "+missing.size());
        System.out.println("Categories:");
        for(Map.Entry<String, Integer> entry : missing.entrySet()) {
            System.out.println(entry.getKey()+" : "+entry.getValue());
        }

        System.out.println("\n\n\n");
    }


    private List<String> tokenizeSentence(String sentence) {
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        Tokenizer t = tokenizerFactory.create(sentence);
        List<String> tokens = new ArrayList<>();
        while (t.hasMoreTokens()) {
            String token = t.nextToken().toLowerCase();
            tokens.add(token);
        }
        return tokens;
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

    public HashMap<String, Integer> learnWord2Id(List<Record> records, int vocabSize) {
        Record record;
        HashMap<String, Integer> wordFreq = Maps.newHashMap();
        for (int i = 0; i < records.size(); i++) {
            record = records.get(i);
            List<String> tokens = tokenizeSentence(record.getLibelle());
            for (String word : tokens) {
                if (wordFreq.containsKey(word)) {
                    wordFreq.put(word, wordFreq.get(word)+1);
                } else {
                    wordFreq.put(word, 1);
                }
            }
        }

        HashMap<String, Integer> word2id = Maps.newHashMap();
        word2id.put("<unk>", word2id.size());
        for(Map.Entry<String, Integer> entry : wordFreq.entrySet()) {
            if (entry.getValue() > 1) {
                word2id.put(entry.getKey(), word2id.size());
            }
        }

        return word2id;
    }



}
