package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import dl.paragraph.acceptance.AcceptanceComparative;
import dl.paragraph.acceptance.AcceptanceType;
import dl.paragraph.my.MySentenceFromListIterator;
import dl.paragraph.my.MySentenceIterator;
import dl.paragraph.my.MySentenceIteratorConverter;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Categorie;
import dl.paragraph.pojo.Record;
import dl.paragraph.pojo.Resultat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Categorisation {

    public static List<String> CATEGORIES = Apollon.labels();
    private ParagraphVectors paragraphVectors;
    private boolean avecMontant = false;
    private boolean avecType = false; // Inclus le type de transaction dans le libellé (RECETTE ou DEPENSE)

    public Categorisation() {
    }


    public void train() throws IOException, InterruptedException {

        // Fichier de la forme : CATEGORIE,LIBELLE,MONTANT
        LabelAwareIterator iterator = readCSVDataset("apollon_data_2018.normalise.csv", avecMontant, avecType);

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        paragraphVectors = ConfigsTests.config2b(iterator, tokenizerFactory);

        // Start model training
        System.out.println("Train starting...");
//        Stopwatch started = Stopwatch.createStarted();
        paragraphVectors.fit();
//        System.out.println("Train done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");
        System.out.println("Train done");

        int vectorSize = paragraphVectors.getWordVector(paragraphVectors.vocab().wordAtIndex(0)).length;   //Size of the word vectors
        int vocabSite = paragraphVectors.vocab().vocabWords().size();   //Size of the word vectors
        System.out.println("Vocab : " + vocabSite);
        System.out.println("Vector : " + vectorSize);

        saveModel(paragraphVectors);

//        //Set up network configuration
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .updater(new Adam()) // 5e-3
//                .l2(1e-5)
//                .weightInit(WeightInit.XAVIER)
//                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
//                .list()
//                .layer(0, new LSTM.Builder().nIn(vectorSize).nOut(256)
//                        .activation(Activation.TANH).build())
//                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
//                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
//                .build();
//
//        MultiLayerNetwork net = new MultiLayerNetwork(conf);
//        net.init();
//        net.setListeners(new ScoreIterationListener(1));
//
    }

    public void evaluate() throws Exception {

        // Fichier de la forme : CATEGORIE,LIBELLE,MONTANT
        evaluate(readCSVRecordsTest("apollon_data_2018.test_small.csv"));
    }

    public void evaluate(List<Record> records) throws Exception {

        if (paragraphVectors == null) {
            paragraphVectors = readModelFromFile();
        }

        if (CATEGORIES == null) {
            throw new Exception("Il faut d'abord renseigner la liste complète des catégories pour évaluer");
        }

        LabelAwareIterator iterator = readDataForEvaluation(records);

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);

        LabelSeeker seeker = new LabelSeeker(
                CATEGORIES,
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        Stopwatch started = Stopwatch.createStarted();
        System.out.println("Evaluation starting...");

//        int limit = 10;
//        while (iterator.hasNextDocument() && limit > 0) {
//            LabelledDocument document = iterator.nextDocument();
//            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
//            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
//            System.out.println("Item [" + document.getContent() + "] falls into the following categories: ");
//            for (int i = 0; i < scores.size(); i++) {
//                Pair<String, Double> score = scores.get(i);
//                System.out.println("        " + score.getFirst() + ": " + score.getSecond());
//            }
//            limit--;
//        }
//        iterator.reset();

        AcceptanceType acceptanceType = new AcceptanceType(new AcceptanceComparative());

        int nbEvaluated = 0;
        List<Resultat> resultats = Lists.newArrayList();
        while (iterator.hasNextDocument()) {

            nbEvaluated++;
            if (nbEvaluated % 1000 == 0) {
                System.out.println("...still going : " + nbEvaluated + " records evaluated in " + started.elapsed(TimeUnit.MILLISECONDS) + "ms");
            }

            LabelledDocument document = iterator.nextDocument();
            Record record = ((MySentenceIteratorConverter) iterator).record();

            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            Resultat resultat = null;
            if (scores != null) {
                Pair<String, Double> score1 = scores.get(0);
                Pair<String, Double> score2 = scores.get(1);
                Pair<String, Double> score3 = scores.get(2);
                resultat = Resultat.builder()
                        .libelle(document.getContent())
                        .record(record)
                        .categorie1(Categorie.builder().label(score1.getFirst()).score(score1.getSecond()).build())
                        .categorie2(Categorie.builder().label(score2.getFirst()).score(score2.getSecond()).build())
                        .categorie3(Categorie.builder().label(score3.getFirst()).score(score3.getSecond()).build())
                        .build();
                resultat.setLabelAccepted(acceptanceType.accept(resultat));
            } else {
                resultat = Resultat.builder()
                        .libelle(document.getContent())
                        .record(record)
                        .categorie1(null)
                        .categorie2(null)
                        .categorie3(null)
                        .build();
            }

            resultats.add(resultat);
        }
        System.out.println("Evaluation done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");

        Tracing.saveResultats(resultats);
        Tracing.saveResultatsParCategorie(resultats);
    }

    public void saveModel(ParagraphVectors model) {
        String path = "./model/paragraph_model.bin";
        WordVectorSerializer.writeParagraphVectors(model, path);
    }

    private ParagraphVectors readModelFromFile() throws IOException {
        File file = new File("model/paragraph_model.bin");
        if (file.exists()) {
            return WordVectorSerializer.readParagraphVectors(file);
        }
        return null;
    }

    /**
     * Lecture d'un fichier CSV formatté (si formatage change, cette fonction doit changer).
     * Transformation des données du csv à la lecture pour une meilleure interprétation.
     * Création d'un iterator de dataSet sur les données.
     * Iterator perso qui gère les données texte et se charge de créer les dataSet, voir MyDataSetIterator.
     *
     * Fichier de la forme : CATEGORIE,LIBELLE,MONTANT
     *
     * @param csvFileClasspath
     * @param avecMontant Indique si on ajoute le montant au libellé
     * @param avecType Indique si on ajoute le type au libellé
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public LabelAwareIterator readCSVDataset(String csvFileClasspath, boolean avecMontant, boolean avecType) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
//                .addColumnsString("id", "numcba", "montant", "libele", "numcompte", "categorie")
                //Or for convenience define multiple columns of the same type
                .addColumnsString("Categorie", "Libelle")
                .addColumnDouble("Montant")
                .addColumnsString("Compte", "Type")
                .build();

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .build();


        // On doit avoir les 3 informations suivantes dans cet ordre :
        // - Montant
        // - Libellé
        // - Catégorie
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        List<String> categoriesLoaded = getCategories(transformProcessRecordReader);
        LabelsSource labelsSource = new LabelsSource(categoriesLoaded); // On passe la liste des catégories de chaque libellé ordonnée
        MySentenceIterator iterator = new MySentenceIterator(transformProcessRecordReader, avecMontant, avecType);
        LabelAwareIterator myDataSetIterator = new SentenceIteratorConverter(iterator, labelsSource);
//        myDataSetIterator.setCollectMetaData(true);
        return myDataSetIterator;
    }

    /**
     * Retourne la liste des catégories de tous les records.
     *
     * @param transformProcessRecordReader
     * @return
     */
    private static List<String> getCategories(LocalTransformProcessRecordReader transformProcessRecordReader) {
        List<String> categories = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            String cat = transformProcessRecordReader.next().get(0).toString();
            categories.add(cat); // All with duplicates
        }
        transformProcessRecordReader.reset();
        return categories;
    }

    /**
     * Fichier de la forme : CATEGORIE,LIBELLE,MONTANT
     *
     * @param csvFileClasspath
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public List<Record> readCSVRecordsTest(String csvFileClasspath) throws IOException, InterruptedException {

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

    private static LabelAwareIterator readDataForEvaluation(List<Record> records) {

        MySentenceFromListIterator iterator = new MySentenceFromListIterator(records);
        return new MySentenceIteratorConverter(iterator);
    }
}
