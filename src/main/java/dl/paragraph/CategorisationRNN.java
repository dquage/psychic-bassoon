package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
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
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
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
//        public void evaluate(List<Record> records) throws Exception {

        if (wordVectors == null) {
            wordVectors = readModelFromFile();
        }

        if (wordVectors == null) {
            throw new Exception("Unable to load Word Vectors");
        }

        if (CATEGORIES == null) {
            throw new Exception("Il faut d'abord renseigner la liste complète des catégories pour évaluer");
        }

        int batchSize = 64;     //Number of examples in each minibatch
        int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;;   //Size of the word vectors
        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateLibellesToLength = 100;  //Truncate reviews with length (# words) greater than this
        final int seed = 0;     //Seed for reproducibility
        int nbClasses = CATEGORIES.size();

//        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces




        //DataSetIterators for training and testing respectively
        List<Record> records = readCSVRecords("apollon_data_2018.train.csv");
//        List<String> categories = getCategories(records);
        MyDataSetIterator train = new MyDataSetIterator(wordVectors, records,
                CATEGORIES, batchSize, truncateLibellesToLength, false, false);

        records = readCSVRecords("apollon_data_2018.test.csv");
//        List<String> categoriesTest = getCategories(records);
        MyDataSetIterator test = new MyDataSetIterator(wordVectors, records,
                CATEGORIES, batchSize, truncateLibellesToLength, false, false);


        // !!!!!!!!!!!!!!!!
//        nbClasses = categories.size();


        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(5e-3, 0, 0, 0))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .list()
                .layer(0, new LSTM.Builder().nIn(vectorSize).nOut(256)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(nbClasses).build())
                .build();

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



        // Test 1 élément nouveau
        INDArray features = test.loadFeaturesFromString("VIR ALMERYS SAS TP-20181224-ALMERYS TP -62 6152300-0098532001-0073-", truncateLibellesToLength);
        INDArray networkOutput = net.output(features);
        long timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point((int) (timeSeriesLength - 1)));

        System.out.println("\n\n-------------------------------");
        System.out.println("Evaluation du libellé: [VIR ALMERYS SAS TP-20181224-ALMERYS TP -62 6152300-0098532001-0073-]");
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));







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
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
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
