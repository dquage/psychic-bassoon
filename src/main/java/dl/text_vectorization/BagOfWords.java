package dl.text_vectorization;

import com.google.common.collect.Lists;
import dl.BasicCSVClassifier;
import dl.Main;
import dl.categorisation.LibelleWord2Vec;
import dl.categorisation.word2vec.MySimpleSentencePreProcessor;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class BagOfWords {

    private static LibelleWord2Vec modelWords;

    public BagOfWords() {
//        modelWords = new LibelleWord2Vec();
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
//        System.out.println("Creating DataSetIterators");
//        DataSetIterator trainIter = getDataSetIterator("depenses2017.data");
////        DataSetIterator testIter = getDataSetIterator("depenses2017.test");
//        int nbOut = trainIter.getLabels().size();




        List<String> libelles = readCSVLibelle();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceIterator iter = new CollectionSentenceIterator(new MySimpleSentencePreProcessor(), libelles);

        BagOfWordsVectorizer vec = new BagOfWordsVectorizer.Builder()
                .setMinWordFrequency(1)
                .setStopWords(new ArrayList<String>())
                .setTokenizerFactory(t)
                .setIterator(iter)
                .build();

        System.out.println("word2vec fit");
        vec.fit();
        System.out.println("word2vec save");
        saveBag(vec);
        System.out.println("word2vec fin");


        INDArray prement_assurance_maf = vec.transform("virement web");
        System.out.println("Labels used: " + prement_assurance_maf);

//        DataSet vectorize = vec.vectorize("Prélèvement assurance MAF", "CHARGES_FINANCIERES");
//        INDArray featureMatrix = vectorize.getFeatureMatrix();
//        System.out.println(">>> " + featureMatrix);
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

    public static void saveBag(BagOfWordsVectorizer model) throws IOException {
        String path = "./model/bag.bin";
        File saveFile = new File(path);
        if (!saveFile.exists()) {
            saveFile.createNewFile();
        }
        SerializationUtils.saveObject(model, saveFile);
    }

    public static BagOfWordsVectorizer readBag(String path) throws IOException {
        return SerializationUtils.readObject(new File(path));
    }

    private static List<String> readCSVLibelle() throws IOException, InterruptedException {

        List<String> libelles = Lists.newArrayList();
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "NumCompte")
                .addColumnCategorical("Categorie", Arrays.asList(BasicCSVClassifier.CATEGORIES))
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
                .categoricalToInteger("Categorie")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource("depenses2017.data").getFile()));

        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            libelles.add(next.get(0).toString());
        }

        return libelles;
    }


}
