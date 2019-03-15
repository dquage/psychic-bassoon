package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import dl.categorisation.word2vec.MyWord2Vec;
import dl.paragraph.my.MyDataSetIterator;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Record;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Catégorisation neuronale !
 * A partir du modèle WordVectors généré par ParagraphVectors.
 */
public class CategorisationRNN {

    public static List<String> CATEGORIES = Apollon.labels();
    private WordVectors wordVectors;
    private boolean avecMontant = false;
    private boolean avecType = false; // Inclus le type de transaction dans le libellé (RECETTE ou DEPENSE)

    public CategorisationRNN() {
    }

//    public void evaluate() throws Exception {
//
//        // Fichier de la forme : CATEGORIE,LIBELLE,MONTANT
//        evaluate(readCSVRecordsTest("apollon_data_2018.test.csv"));
//    }


    public void evaluate() throws Exception {

        if (wordVectors == null) {
            MyWord2Vec myWord2vec = new MyWord2Vec();
            wordVectors = myWord2vec.getWord2Vec();
        }

        MultiLayerNetwork net = loadModel();
        int batchSize = 1000;     //Number of examples in each minibatch
        int truncateLibellesToLength = 30;  //Truncate reviews with length (# words) greater than this

        // To SAVE //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        List<Record> testRecords  = readCSVRecords("apollon_data_2018.test.csv");
        List<Record> trainRecords = readCSVRecords("apollon_data_2018.train.csv");
        List<Record> allRecords = new ArrayList<>(testRecords);
        allRecords.addAll(trainRecords);
        List<String> labels = getCategories(allRecords);
//        List<String> labels = Arrays.asList(new String[]{"ENTRETIEN ET REPARATIONS", "FRAIS DE TELECOMMUNICATION"});
        // To SAVE //////////////////////////////////////////////////////////////////////////////////////////////////////////////

        int nbLabels = labels.size();
        System.out.println("Le jeu de données a " + nbLabels + " catégories");

        // Test 1 élément nouveau
        MyDataSetIterator test  = new MyDataSetIterator(wordVectors, testRecords, labels, batchSize, truncateLibellesToLength, false, false);

        INDArray features = test.loadFeaturesFromString("VIREMENT ALMERYS SAS TP 20181224 ALMERYS TP");
//        Evaluation eval = new Evaluation(nbLabels);
//        INDArray output = net.output(features);
////        eval.eval(test.loadLabel("COMPTE DE L EXPLOITANT", truncateLibellesToLength), output);
//        eval.eval(features);
//        System.out.println(eval.stats());

        INDArray networkOutput = net.output(features);
        System.out.println("Evaluation du libellé : [VIREMENT ALMERYS SAS TP 20181224 ALMERYS TP]");
        for (int i = 0; i < labels.size(); i++) {
            System.out.println("[" + labels.get(i) + "] " + networkOutput.getDouble(i));
        }


//        long timeSeriesLength = truncateLibellesToLength;
//        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point((int) (timeSeriesLength - 1)));
//
//        System.out.println("\n\n-------------------------------");
//        System.out.println("Evaluation du libellé: [VIREMENT ALMERYS SAS TP 20181224 ALMERYS TP]");
//        System.out.println("\n\nProbabilities at last time step:");
//        System.out.println("");
//        for (int i = 0; i < nbLabels; i++) {
//            System.out.print("[" + labels.get(i) + "][" + probabilitiesAtLastWord.getDouble(i) + "]");
//        }
//        System.out.println("");




//        Stopwatch started = Stopwatch.createStarted();
//        System.out.println("Evaluation starting...");
//
//        AcceptanceType acceptanceType = new AcceptanceType(new AcceptanceComparative());
//
//        int nbEvaluated = 0;
//        List<Resultat> resultats = Lists.newArrayList();
//        while (iterator.hasNextDocument()) {
//
//            nbEvaluated++;
//            if (nbEvaluated % 1000 == 0) {
//                System.out.println("...still going : " + nbEvaluated + " records evaluated in " + started.elapsed(TimeUnit.MILLISECONDS) + "ms");
//            }
//
//            LabelledDocument document = iterator.nextDocument();
//            Record record = ((MySentenceIteratorConverter) iterator).record();
//
//            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
//            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
//
//            Resultat resultat = null;
//            if (scores != null) {
//                Pair<String, Double> score1 = scores.get(0);
//                Pair<String, Double> score2 = scores.get(1);
//                Pair<String, Double> score3 = scores.get(2);
//                resultat = Resultat.builder()
//                        .libelle(document.getContent())
//                        .record(record)
//                        .categorie1(Categorie.builder().label(score1.getFirst()).score(score1.getSecond()).build())
//                        .categorie2(Categorie.builder().label(score2.getFirst()).score(score2.getSecond()).build())
//                        .categorie3(Categorie.builder().label(score3.getFirst()).score(score3.getSecond()).build())
//                        .build();
//                resultat.setLabelAccepted(acceptanceType.accept(resultat));
//            } else {
//                resultat = Resultat.builder()
//                        .libelle(document.getContent())
//                        .record(record)
//                        .categorie1(null)
//                        .categorie2(null)
//                        .categorie3(null)
//                        .build();
//            }
//
//            resultats.add(resultat);
//        }
//        System.out.println("Evaluation done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
//
//        Tracing.saveResultats(resultats);
//        Tracing.saveResultatsParCategorie(resultats);
    }



    /**
     * 1er essai avec un iterator maison.
     * >>> IllegalStateException: Mis matched shapes
     *
     * @throws Exception
     */
    public void train_and_evaluate() throws Exception {
//        public void evaluate(List<Record> records) throws Exception {

        if (wordVectors == null) {
//            wordVectors = readModelFromFile();
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

        printDataStats(trainRecords, testRecords);

        List<String> labels = getCategories(allRecords);
//        List<String> labels = Arrays.asList(new String[]{"ENTRETIEN ET REPARATIONS", "FRAIS DE TELECOMMUNICATION"});
//        Collections.sort(labels); // FIXME A voir
        int nbClasses = labels.size();
        System.out.println("Num. classes : "+nbClasses);

        int batchSize = 1000;     //Number of examples in each minibatch
        int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;   //Size of the word vectors
        int vobabSize = wordVectors.vocab().numWords();
        System.out.println("Vector size : "+vectorSize);
        System.out.println("Vocab size : "+vobabSize);

        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateLibellesToLength = 30;  //Truncate reviews with length (# words) greater than this
        final int seed = 0;     //Seed for reproducibility




//        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces


        // https://www.youtube.com/watch?v=2-Ol7ZB0MmU

        MyDataSetIterator train = new MyDataSetIterator(wordVectors, trainRecords, labels, batchSize, truncateLibellesToLength, false, false);
        MyDataSetIterator test  = new MyDataSetIterator(wordVectors, testRecords, labels, batchSize, truncateLibellesToLength, false, false);

//        printDataStats(trainRecords, testRecords);
//
//        System.out.println("-----------  Concordance  ------------------");
//        String cat;
//        cat = "PART PRIVEE DES DEPENSES MIXTES"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "HONORAIRES"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "FOURNITURES DE BUREAU ET ADMINISTRATIVES"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "COTISATIONS COMPLEMENTAIRES SANTE"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));
//        cat = "FRAIS DE FORMATION SEMINAIRE"; System.out.println(train.getLabelsClassMap().get(cat)+" <=> "+test.getLabelsClassMap().get(cat));



//        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(0.001D, 0.9D, 0.999D, 1.0E-8D))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .list()
                .layer(0, new LSTM.Builder().nIn(vectorSize).nOut(256).activation(Activation.TANH).build())
//                .layer(1, new DenseLayer.Builder().nIn(256).nOut(256).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
                .layer(1, new RnnOutputLayer.Builder().nIn(256).nOut(nbClasses).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();



        // !!!!!!!!!!!!!!!!
//        nbClasses = categories.size();

//        this.learningRate = 0.001D;

//        //Set up network configuration
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
//                .layer(0, new DenseLayer.Builder().nIn(truncateLibellesToLength).nOut(30).weightInit(WeightInit.XAVIER).activation(Activation.RELU) //First hidden layer
//                        .build())
//                .layer(1, new OutputLayer.Builder().nIn(30).nOut(nbClasses).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX) //Output layer
//                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .build())
//                .pretrain(false).backprop(true)
//                .build();

        // !!!!!!!!!!!!!!!!
//        nbClasses = categories.size();

////        this.learningRate = 0.001D;
//        //Set up network configuration
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                 .updater(new Adam(0.001D, 0.9D, 0.999D, 1.0E-8D))
//                .l2(1e-5)
//                .weightInit(WeightInit.XAVIER)
//                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
////                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .list()
//                .layer(0, new LSTM.Builder().nIn(vectorSize).nOut(nbClasses*2).activation(Activation.TANH).build())
////                .layer(1, new DenseLayer.Builder().activation(Activation.RELU).nIn(nbClasses*2).nOut(nbClasses).build())
//                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).nIn(nbClasses*2).nOut(nbClasses).build())
//                .build();

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
//                .layer(0, new DenseLayer.Builder().nIn(vectorSize).nOut(nbClasses*2).weightInit(WeightInit.XAVIER).activation(Activation.RELU) //First hidden layer
//                        .build())
//                .layer(1, new OutputLayer.Builder().nIn(nbClasses*2).nOut(nbClasses).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX) //Output layer
//                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .build())
//                .pretrain(false).backprop(true)
//                .build();

        System.out.println("Config built");


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));




        Stopwatch started = Stopwatch.createStarted();
        System.out.println("Training starting...");
        for (int i = 0; i < nEpochs; i++) {
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

        System.out.println("################");
        System.out.println("## Mots non vectorisés par notre modèle wordVectors (" + train.getWordsNonVectorises().size() + ") : ");
        System.out.println(train.getWordsNonVectorises());

        saveModel(net);
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

    private void saveModel(MultiLayerNetwork net) throws IOException {
        File file = new File("model/rnn_model.zip");
        ModelSerializer.writeModel(net, file, true);
    }

    private MultiLayerNetwork loadModel() throws IOException {
        File file = new File("model/rnn_model.zip");
        return ModelSerializer.restoreMultiLayerNetwork(file);
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

    public void saveModel(ParagraphVectors model) {
        String path = "./model/paragraph_model.bin";
        WordVectorSerializer.writeParagraphVectors(model, path);
    }

    private WordVectors readModelFromFile() throws IOException {
        File file = new File("model/paragraph_model.bin");
        if (file.exists()) {
            return WordVectorSerializer.loadStaticModel(file);
        }
        return null;
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

    public HashMap<String, Integer> learnWord2Id(List<Record> records) {
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
