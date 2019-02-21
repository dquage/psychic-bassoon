package dl.paragraph;

import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class ConfigsTests {

    /**
     * Test libellés uniquement
     * Train done In 16s
     * Nombre éléments catégorisés avec succès : 98 donc 29.79%
     */
    public static ParagraphVectors config1(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) {
        return new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    /**
     * Test libellés uniquement
     * Train done In 57s
     * Nombre éléments catégorisés avec succès : 119 donc 36.17%
     *
     * Test libellés + montants
     * Train done In 56s
     * Nombre éléments catégorisés avec succès : 120 donc 36.47%
     */
    public static ParagraphVectors config2(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) {
        return new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .learningRate(0.025)
                .minLearningRate(0.001)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    /**
     * Test libellés uniquement
     * Train done In 53s
     * Nombre éléments catégorisés avec succès : 110 donc 33.43%
     */
    public static ParagraphVectors config2a(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) {
        return new ParagraphVectors.Builder()
                .minWordFrequency(2) // <---- Augmente
                .iterations(5)
                .learningRate(0.025)
                .minLearningRate(0.001)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    /**
     * Test libellés uniquement
     * Train done In 111s
     * Nombre éléments catégorisés avec succès : 120 donc 36.47%
     */
    public static ParagraphVectors config2b(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) {
        return new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(10) // <---- Augmente
                .learningRate(0.025)
                .minLearningRate(0.001)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    /**
     * Test libellés uniquement
     * Train done In 162s
     * Nombre éléments catégorisés avec succès : 128 donc 38.91%
     *
     * Test libellés + montants
     * Train done In 164s
     * Nombre éléments catégorisés avec succès : 123 donc 37.39%
     */
    public static ParagraphVectors config2c(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) {
        return new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(10) // <---- Augmente
                .learningRate(0.025)
                .minLearningRate(0.001)
                .epochs(30) // <---- Augmente
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    /**
     * Test libellés uniquement
     * Train done In 62s
     * Nombre éléments catégorisés avec succès : 111 donc 33.74%
     */
    public static ParagraphVectors config3(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) {
        return new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .epochs(20)
                .layerSize(100)
                .windowSize(5)
                .sampling(0)
                .learningRate(0.025)
                .minLearningRate(0.001)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    /**
     * Test libellés uniquement
     * Train done In 7s
     * Nombre éléments catégorisés avec succès : 61 donc 18.54%
     */
    public static ParagraphVectors config4(LabelAwareIterator iterator, TokenizerFactory tokenizerFactory) {
        return new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .epochs(1)
                .windowSize(5)
                .sampling(0)
                .learningRate(0.025)
                .minLearningRate(0.001)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }
}
