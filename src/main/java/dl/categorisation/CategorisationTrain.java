package dl.categorisation;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import dl.categorisation.iterator.MyDataSetIterator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.iterator.Word2VecDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Apprentissage du modèle pour la catégorisation.
 *
 * Utilisation des données modèles de APOLLON (format csv).
 * Utilisation de notre dictionnaire de texte généré word2vec pour transformer les libellés en vectors.
 * Evaluation du modèle et enregistrement pour utilisation directe.
 *
 */
public class CategorisationTrain {

    public static String[] CATEGORIES = {
            "SERVICES_BANCAIRES", "COMPTE_DE_L_EXPLOITANT", "DEPLACEMENTS_MISSIONS_RECEPTIONS", "REDEVANCE_COLLABORATION", "ESSENCE_VEHICULE",
            "CHARGES_DIVERSES_DE_GESTION_COURANTE", "REPAS_DE_MIDI", "ENTRETIEN_ET_REPARATIONS", "DEPENSES_VEHICULE",
            "EMPRUNT", "COTISATIONS_COMPLEMENTAIRES_EXPLOITANT", "CHARGES_FINANCIERES", "ASSURANCE_VEHICULE", "FRAIS_DE_TELECOMMUNICATION",
            "HONORAIRES_RETROCEDES", "FOURNITURES_DE_BUREAU_ET_ADMINISTRATIVES", "PETIT_OUTILLAGE_PETIT_MATERIEL", "ASSURANCES",
            "COTISATIONS_PROF_ET_SYNDICALE", "FRAIS_POSTAUX", "CFE_CVAE", "ACHATS_A_USAGE_UNIQUE", "TRANSPORTS_DIVERS__BUS_TAXIS_",
            "HONORAIRES_RETROCEDES_A_REMPLACANT", "CADEAUX_CLIENTELE", "COTISATION_CARPIMKO", "FRAIS_DE_FORMATION_SEMINAIRE",
            "CHEQUES_VACANCES", "FRAIS_DE_DOCUMENTATION_TECHNIQUE", "LOCATIONS_IMMOBILIERES", "LLD_VEHICULE", "AUTRES_SERVICES_EXTERIEURS_HONORAIRES",
            "HONORAIRES", "AGIOS_BANCAIRES", "PRESTATIONS_DE_SERVICE", "COTISATIONS_MALADIE_EXPLOITANT",
            "COMPTE_ASSOCIE", "AUTRES_IMPOTS", "LEASING_AUTRE_QUE_VEHICULE", "EDF_GDF_CHAUFFAGE", "REMUNERATIONS_DU_PERSONNEL",
            "COTISATIONS_URSSAF_DE_L_EXPLOITANT", "PRODUITS_D_ENTRETIEN", "LEASING_VEHICULE", "COTISATIONS_COMPLEMENTAIRES_SANTE", "PART_PRIVEE_DES_DEPENSES_MIXTES",
            "PERTE_EMPLOI_MADELIN", "FRAIS_DE_CONGRES", "LOCATION_LD_AUTRE_QUE_VEHICULE", "COTISATIONS_COMPLEMENTAIRES_RETRAITE",
            "FRAIS_D_ACTES_ET_DE_CONTENTIEUX", "FOURNISSEURS_D_IMMOBILISATIONS", "ACHATS_MARCHANDISES", "COMPTE_D_ATTENTE", "APPORT_PERSONNEL",
            "SECURITE_SOC_ET_PREVOYANCE_DU_PERSONNEL", "COTISATIONON_CARPIMKO", "CREDIT_BAIL_LEASING_MOBILIER"
    };

    private MultiLayerNetwork model;
    private LibelleWord2Vec modelWords;
    private int batchSize = 64;     //Number of examples in each minibatch
    private int seed = 0;     //Seed for reproducibility
    private int nEpochs = 5;
    private int nbIn_features_count = 1; // Nombre de lignes de données pour le train (taille max du vecteur de texte)
    private int nbOut_classes_count = 1; // Nombre de colonnes utilisées pour la classification

    public CategorisationTrain(LibelleWord2Vec modelWords) {
        this.modelWords = modelWords;
//        vectorSize = modelWords.getVecLength();
    }


    @SuppressWarnings("unchecked")
    public void train() throws Exception {

        DataSetIterator iter = readCSVDataset("depenses2017.data", modelWords, batchSize);

        nbIn_features_count = 100;
        nbOut_classes_count = 1;

        System.out.println("Build model....");

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Nesterovs(0.01, 0.9))
//                .list()
//                .layer(0, new DenseLayer.Builder().nIn(nbIn_features_count).nOut(20)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
//                        .activation(Activation.SIGMOID)
//                        .nIn(20).nOut(nbOut_classes_count).build())
//                .build();

        // Modèle généré à 3Ko mais pas de platage à l'évaluation
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(100)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.01)
                .regularization(true)
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nbIn_features_count).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2,
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX).nIn(3).nOut(nbOut_classes_count).build())
                .backprop(true)
                .pretrain(false)
                .build();

        // Modèle généré à 3Ko mais pas de platage à l'évaluation
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .activation(Activation.TANH)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Sgd(0.1))
//                .l2(1e-4)
//                .list()
//                .layer(0, new DenseLayer.Builder().nIn(nbIn_features_count).nOut(3).build())
//                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
//                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.SOFTMAX).nIn(3).nOut(nbOut_classes_count).build())
//                .backprop(true).pretrain(false)
//                .build();

//        MultiLayerConfiguration conf =  new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(10)
//                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                .gradientNormalizationThreshold(1.0)
//                .regularization(true)
//                .dropOut(0.5)
//                .updater(Updater.ADADELTA)
////                .adamMeanDecay(0.5)
////                .adamVarDecay(0.5)
//                .weightInit(WeightInit.XAVIER)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .list()
//                .layer(0, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.GAUSSIAN)
//                        .nIn(nbIn_features_count).nOut(2750).dropOut(0.75)
//                        .activation(Activation.RELU).build())
//                .layer(1, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
//                        .nIn(2750).nOut(2000)
//                        .activation(Activation.RELU).build())
//                .layer(2, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
//                        .nIn(2000).nOut(1000)
//                        .activation(Activation.RELU).build())
//                .layer(3, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
//                        .nIn(1000).nOut(200)
//                        .activation(Activation.RELU).build())
//                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
//                        .nIn(200).nOut(nbOut_classes_count).updater(Updater.ADADELTA)
////                        .adamMeanDecay(0.6).adamVarDecay(0.7)
//                        .build())
//                .pretrain(true).backprop(true)
//                .build();

        // Erreur Mis matched shapes
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(10)
////                .updater(new Adam(5e-3, 0, 0, 0))
//                .updater(Updater.ADAM)
//                .l2(1e-5)
//                .weightInit(WeightInit.XAVIER)
//                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
//                .list()
//                .layer(0, new LSTM.Builder().nIn(nbIn_features_count).nOut(256)
//                        .activation(Activation.TANH).build())
//                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
//                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(nbOut_classes_count).build())
//                .build();


        // Erreur Mis matched shapes
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .iterations(10)
//                .updater(Updater.ADADELTA)
//                .l2(1e-5)
//                .weightInit(WeightInit.XAVIER)
//                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
//                .list()
//                .layer(0, new LSTM.Builder().nIn(nbIn_features_count).nOut(256).activation(Activation.TANH).build())
//                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX).nIn(256).nOut(nbOut_classes_count).build())
//                .build();

        conf.setPretrain(false);

        //run the model
        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        Stopwatch started = Stopwatch.createStarted();
        System.out.println("Train model....");

//        for (int i = 0; i < nEpochs; i++) {
            model.fit(iter);
//            System.out.println("Epoch " + i + " complete. Starting evaluation:");
//        }

        System.out.println(model.summary());
        System.out.println();
        System.out.println("Training done. In " + started.stop().elapsed(TimeUnit.SECONDS) + "s");

        save(model, "model/model.zip");
    }

    /**
     * Chargement des données de test.
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private DataSetIterator prepareEvaluation() throws IOException, InterruptedException {

        // Chargement du modèle enregistré + Chrono
        if (model == null) {
            Stopwatch started = Stopwatch.createStarted();
            model = loadModel("model/model.zip");
            System.out.println("Loading model done. In " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
        }

        // Lecture des données de test, obtention d'un iterator sur toutes les données
        return readCSVDataset("depenses2017.test", modelWords, batchSize);
    }

    /**
     * Evaluation du modèle à partir des données de test.
     * @throws IOException
     * @throws InterruptedException
     */
    public void evaluate() throws IOException, InterruptedException {

        DataSetIterator iter = prepareEvaluation();

        // Lancement de l'évaluation des données de test
        Stopwatch started = Stopwatch.createStarted();
        System.out.println("Start evaluating");

        Evaluation evaluation = model.evaluate(iter);

        System.out.println(evaluation.stats());

//        Evaluation eval = new Evaluation(CATEGORIES.length);
//        while (iter.hasNext()) {
//            DataSet next = iter.next();
//            INDArray output = model.output(next.getFeatures());
//            eval.eval(next.getLabels(), output);
//            System.out.println(eval.stats());
//        }

        System.out.println("Evaluating done In " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
        System.out.println();
    }

    /**
     * Test évaluation par régression.
     * Utile ?
     * @throws IOException
     * @throws InterruptedException
     */
    public void regevaluation() throws IOException, InterruptedException {

        DataSetIterator iter = prepareEvaluation();

        RegressionEvaluation eval = new RegressionEvaluation(nbOut_classes_count);
        while(iter.hasNext()) {
            DataSet ds = iter.next();
            ds.shuffle();
            INDArray output = model.output(ds.getFeatureMatrix());
            eval.eval(ds.getLabels(), output);
        }
        System.out.println("Testing done");
        System.out.println(eval.stats());
    }

    /**
     * Enregistrement du modèle.
     * @param model
     * @param path
     * @throws IOException
     */
    private static void save(MultiLayerNetwork model, String path) throws IOException {

        File file = new File(path);
        if (!file.exists()) {
            file.createNewFile();
        }
        ModelSerializer.writeModel(model, file, false);
    }

    /**
     * Chargement du modèle enregistré.
     * @param path
     * @return
     * @throws IOException
     */
    private static MultiLayerNetwork loadModel(String path) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(path);
    }

    /**
     * Transforme le tableau des catégories en Map pour correspondance en int.
     * @return
     */
    public static Map<Integer, String> readClassifier() {
        String[] classes = CATEGORIES;
        Map<Integer, String> enums = new HashMap<>();
        for (int i = 0; i < classes.length; i++) {
            enums.put(i + 1, classes[i]);
        }
        return enums;
    }

    /**
     * Lecture d'un fichier CSV formatté (si formatage change, cette fonction doit changer).
     * Transformation des données du csv à la lecture pour une meilleure interprétation.
     * Création d'un iterator de dataSet sur les données.
     * Iterator perso qui gère les données texte et se charge de créer les dataSet, voir MyDataSetIterator.
     *
     * @param csvFileClasspath
     * @param modelWords
     * @param batchSize
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSetIterator readCSVDataset(String csvFileClasspath, LibelleWord2Vec modelWords, int batchSize) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
//                .addColumnsString("id", "numcba", "montant", "libele", "numcompte", "categorie")
                //Or for convenience define multiple columns of the same type
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "NumCompte")
                .addColumnCategorical("Categorie", Arrays.asList(CATEGORIES))
                .build();

        //Print out the schema:
//        System.out.println("Input data schema details:");
//        System.out.println(inputDataSchema);
//
//        System.out.println("\n\nOther information obtainable from schema:");
//        System.out.println("Number of columns: " + inputDataSchema.numColumns());
//        System.out.println("Column names: " + inputDataSchema.getColumnNames());
//        System.out.println("Column types: " + inputDataSchema.getColumnTypes());

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Id", "NumCba", "Montant", "NumCompte")
//                .categoricalToInteger("Categorie")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);
        MyDataSetIterator myDataSetIterator = new MyDataSetIterator(transformProcessRecordReader, batchSize, 1, 0, modelWords);
//        myDataSetIterator.setCollectMetaData(true);
        return myDataSetIterator;
    }

    /**
     * Test sentence iterator, non concluant.
     * @param csvFileClasspath
     * @param modelWords
     * @param batchSize
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static Word2VecDataSetIterator readCSVDataset2(String csvFileClasspath, LibelleWord2Vec modelWords, int batchSize) throws IOException, InterruptedException {

        LabelAwareListSentenceIterator libIter = new LabelAwareListSentenceIterator(new FileInputStream(new ClassPathResource(csvFileClasspath).getFile()), ",", 5, 3);
        Word2VecDataSetIterator iter = new Word2VecDataSetIterator(modelWords.getLoadedVec(), libIter, Arrays.asList(CATEGORIES), batchSize);
        return iter;
    }
}
