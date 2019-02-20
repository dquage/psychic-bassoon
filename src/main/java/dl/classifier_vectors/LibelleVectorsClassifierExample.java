package dl.classifier_vectors;

import au.com.bytecode.opencsv.CSVWriter;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;

/**
 * This is basic example for documents classification done with DL4j ParagraphVectors.
 * The overall idea is to use ParagraphVectors in the same way we use LDA:
 * topic space modelling.
 *
 * In this example we assume we have few labeled categories that we can use
 * for training, and few unlabeled documents. And our goal is to determine,
 * which category these unlabeled documents fall into
 *
 *
 * Please note: This example could be improved by using learning cascade
 * for higher accuracy, but that's beyond basic example paradigm.
 *
 * @author raver119@gmail.com
 */
public class LibelleVectorsClassifierExample {

    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    public void makeParagraphVectors()  throws Exception {

        ClassPathResource resource = new ClassPathResource("test"); // Lira tous les fichiers de ce répertoire situé dans resources
        System.out.println(resource.getFile().getPath());




        SentenceIterator iter = new BasicLineIterator(resource.getFile());

        TokenizerFactory tFact = new DefaultTokenizerFactory();
        tFact.setTokenPreProcessor(new CommonPreprocessor());
        LabelsSource labelFormat = new LabelsSource("LINE_"); // Préfixe chaque ligne avec LINE_

        paragraphVectors = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .layerSize(100)
                .epochs(1)
                .labelsSource(labelFormat)
                .windowSize(5)
                .sampling(0)

                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tFact)
                .build();









                // build a iterator for our dataset
        iterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(resource.getFile())
                .build();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();

//        Word2Vec.Builder()
//                .seed(12345)
//                .iterate(iter)
//                .tokenizerFactory(t)
//                .batchSize(1000)
//                .allowParallelTokenization(true) // enable parallel tokenization
//                .epochs(1) //  number of epochs (iterations over whole training corpus) for training
//                .iterations(3) // number of iterations done for each mini-batch during training
//                .elementsLearningAlgorithm(new SkipGram<>()) // use SkipGram Model. If CBOW: new CBOW<>()
//                .minWordFrequency(50) // discard words that appear less than the times of set value
//                .windowSize(5) // set max skip length between words
//                .learningRate(0.05) // the starting learning rate
//                .minLearningRate(5e-4) // learning rate should not lower than the set threshold value
//                .negativeSample(10) // number of negative examples
//                // set threshold for occurrence of words. Those that appear with higher frequency will be
//                // randomly down-sampled
//                .sampling(1e-5)
//                .useHierarchicSoftmax(true) // use hierarchical softmax
//                .layerSize(300) // size of word vectors
//                .workers(8) // number of threads
//                .build();

        // Start model training
        paragraphVectors.fit();
    }

    private void writeIndexToCsv(String csvFileName, Word2Vec model) {

        CSVWriter writer = null;
        try {
            writer = new CSVWriter(new FileWriter(csvFileName));
        } catch (IOException e) {
            e.printStackTrace();
        }

        VocabCache<VocabWord> vocCache =  model.vocab();
        Collection<VocabWord> wrds = vocCache.vocabWords();

        for(VocabWord w : wrds) {
            String s = w.getWord();
            System.out.println("Looking into the word:");
            System.out.println(s);
            StringBuilder sb = new StringBuilder();
            sb.append(s).append(",");
            double[] wordVector = model.getWordVector(s);
            for(int i = 0; i < wordVector.length; i++) {
                sb.append(wordVector[i]).append(",");
            }

            writer.writeNext(sb.toString().split(","));
        }

        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
