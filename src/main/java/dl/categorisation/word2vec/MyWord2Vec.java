package dl.categorisation.word2vec;


import com.google.common.collect.Lists;
import dl.categorisation.CategorisationTrain;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class MyWord2Vec {

    private static final Integer VEC_LENGTH = 100;

    private Word2Vec word2Vec;

    public MyWord2Vec() throws IOException, InterruptedException {
        word2Vec = readModelFromFile();
        if (word2Vec == null) {
            word2Vec = createWord2Vec();
        }
    }

    private Word2Vec createWord2Vec() throws IOException, InterruptedException {

        List<String> libelles = readCSVLibelle();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        ElementsLearningAlgorithm<VocabWord> learningAlgorithm = new SkipGram<>();

        System.out.println("word2vec apprentissage");
        SentenceIterator iter = new CollectionSentenceIterator(new MySimpleSentencePreProcessor(), libelles);
        Word2Vec vec = new Word2Vec.Builder()
                .elementsLearningAlgorithm(learningAlgorithm)
                .minWordFrequency(5)
                .iterations(15)
                .layerSize(VEC_LENGTH)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        System.out.println("word2vec fit");
        vec.fit();
        System.out.println("word2vec save");
        saveModel(vec);
        System.out.println("word2vec fin");


        // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
        System.out.println("Test closest words:");
        Collection<String> lst = vec.wordsNearestSum("assurance", 10);
        System.out.println("10 Words closest to 'day': " + lst);

        return vec;
    }

    public void saveModel(Word2Vec model) {
        String path = "./model/word2vec_model.bin";
        WordVectorSerializer.writeWord2VecModel(model, path);
    }

    private Word2Vec readModelFromFile() {
        File file = new File("model/word2vec_model.bin");
        if (file.exists()) {
            return WordVectorSerializer.readWord2VecModel(file);
        }
        return null;
    }

    private List<String> readCSVLibelle() throws IOException, InterruptedException {

        List<String> libelles = Lists.newArrayList();
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "NumCompte")
                .addColumnCategorical("Categorie", Arrays.asList(CategorisationTrain.CATEGORIES))
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

    public Word2Vec getWord2Vec() {
        return word2Vec;
    }
}
