package dl;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class BasicCSVClassifier {

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

    private static Map<Integer,String> classifiers = readClassifier(); // Enum des classes (catégories) de classement map en int pour le process

    public static void test(String filenameData) throws IOException, InterruptedException {

        int categorieIndex = 4;     // Position de la catégorie
        int numClasses = CATEGORIES.length;

        int batchSize = 64;     //Number of examples in each minibatch
        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this
        final int seed = 0;     //Seed for reproducibility

        DataSet allData = readCSVDataset(filenameData, batchSize, categorieIndex, numClasses);
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.70);  //Use 70% of data for training

        // this is the data we want to classify
//        int batchSizeTest = 44;
//        DataSet testData = readCSVDataset(filenameData, batchSizeTest, categorieIndex, numClasses);

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();




        //DataSetIterators for training and testing respectively
//        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
//        SentimentExampleIterator train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
//        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);



        // make the data model for records prior to normalization, because it
        // changes the data.
        Map<Integer, Map<String,Object>> items = makeRecordModel(testData);


        if (true) return;

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        final int numInputs = 4;
        int outputNum = 3;

        System.out.println("Build model....");
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .activation(Activation.TANH)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Sgd(0.1))
//                .l2(1e-4)
//                .list()
//                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
//                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
//                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
//                .backprop(true).pretrain(false)
//                .build();

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
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));
//        model.setListeners(new ScoreIterationListener(100));


//        for (int i = 0; i < nEpochs; i++) {
//            model.fit(trainingData);
//        }
//
//        //evaluate the model on the test set
//        Evaluation eval = new Evaluation(3);
//        INDArray output = model.output(testData.getFeatures());
//
//        eval.eval(testData.getLabels(), output);
//        System.out.println(eval.stats());
//
//        setFittedClassifiers(output, items);
//        logAnimals(items);


//        System.out.println("Starting training");
//        for (int i = 0; i < nEpochs; i++) {
//            model.fit(train);
//            train.reset();
//            System.out.println("Epoch " + i + " complete. Starting evaluation:");
//
//            //Run evaluation. This is on 25k reviews, so can take some time
//            Evaluation evaluation = model.evaluate(test);
//            System.out.println(evaluation.stats());
//        }

        //After training: load a single example and generate predictions
//        File firstPositiveReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/pos/0_10.txt"));
//        String firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile);
//
//        INDArray features = test.loadFeaturesFromString(firstPositiveReview, truncateReviewsToLength);
//        INDArray networkOutput = net.output(features);
//        long timeSeriesLength = networkOutput.size(2);
//        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));
//
//        System.out.println("\n\n-------------------------------");
//        System.out.println("First positive review: \n" + firstPositiveReview);
//        System.out.println("\n\nProbabilities at last time step:");
//        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
//        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));
//
//        System.out.println("----- Example complete -----");





        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Early stopping pour avoir un apprentissage optimisé, faut-il encore que ça marche
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        DataSetIterator iterTest = readCSVDataset("depenses2017.test", modelWords, batchSize);
//
//        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
//                .epochTerminationConditions(new MaxEpochsTerminationCondition(30))
//                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(4, TimeUnit.MINUTES))
//                .scoreCalculator(new DataSetLossCalculator(iterTest, true))
//                .evaluateEveryNEpochs(1)
//                .modelSaver(new LocalFileModelSaver("models"))
//                .build();
//
//        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, iter);
//
//        //Conduct early stopping training:
//        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
//
//        //Print out the results:
//        System.out.println("Termination reason: " + result.getTerminationReason());
//        System.out.println("Termination details: " + result.getTerminationDetails());
//        System.out.println("Total epochs: " + result.getTotalEpochs());
//        System.out.println("Best epoch number: " + result.getBestModelEpoch());
//        System.out.println("Score at best epoch: " + result.getBestModelScore());
//
//        //Get the best model:
//        model = result.getBestModel();
//        if (bestModel == null) {
//            throw new Exception("Best model not determinated");
//        }
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//        System.out.println(eval.stats());
    }

    public static void logAnimals(Map<Integer, Map<String, Object>> animals) {
        for (Map<String, Object> a : animals.values())
            System.out.println(a.toString());
    }

    public static void setFittedClassifiers(INDArray output, Map<Integer, Map<String, Object>> animals) {
        for (int i = 0; i < output.rows(); i++) {
            // set the classification from the fitted results
            animals.get(i).put("classifier", classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));
        }
    }

    /**
     * This method is to show how to convert the INDArray to a float array. This is to
     * provide some more examples on how to convert INDArray to types that are more java
     * centric.
     *
     * @param rowSlice
     * @return
     */
    public static float[] getFloatArrayFromSlice(INDArray rowSlice){
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * find the maximum item index. This is used when the data is fitted and we
     * want to determine which class to assign the test row to
     *
     * @param vals
     * @return
     */
    public static int maxIndex(float[] vals){
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++){
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * take the dataset loaded for the matric and make the record model out of it so
     * we can correlate the fitted classifier to the record.
     *
     * @param testData
     * @return
     */
    public static Map<Integer, Map<String, Object>> makeRecordModel(DataSet testData) {

        Map<Integer, Map<String, Object>> items = new HashMap<>();
        INDArray features = testData.getFeatures();
        for (int i = 0; i < features.rows(); i++) {
            INDArray slice = features.slice(i);
            Map<String, Object> item = new HashMap<>();

            //set the attributes TODO
            item.put("Id", slice.getInt(0));
            item.put("Montant", slice.getInt(2));

            items.put(i, item);
        }
        return items;
    }

    public static Map<Integer,String> readClassifier() {
        String[] classes = CATEGORIES;
        Map<Integer, String> enums = new HashMap<>();
        for (int i = 0; i < classes.length; i++) {
            enums.put(i+1, classes[i]);
        }
        return enums;
    }

//    public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
//        try{
//            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
//            Map<Integer,String> enums = new HashMap<>();
//            for(String line:lines){
//                String[] parts = line.split(",");
//                enums.put(Integer.parseInt(parts[0]),parts[1]);
//            }
//            return enums;
//        } catch (Exception e){
//            e.printStackTrace();
//            return null;
//        }
//    }

    /**
     * used for testing and training
     *
     * @param csvFileClasspath
     * @param batchSize
     * @param labelIndex
     * @param numClasses
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses) throws IOException, InterruptedException {

//        RecordReader rr = new CSVRecordReader();
//        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
//        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);


        Schema inputDataSchema = new Schema.Builder()
//                .addColumnsString("id", "numcba", "montant", "libele", "numcompte", "categorie")
                //Or for convenience define multiple columns of the same type
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "NumCompte")
                .addColumnCategorical("Categorie", Arrays.asList(CATEGORIES))
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
                .removeColumns("Id", "NumCba", "NumCompte")
                .convertToInteger("Libelle")
                .categoricalToInteger("Categorie")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
//        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);
        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader, labelIndex, numClasses, batchSize);

        return iterator.next();
    }
}
