package dl.categorisation.word2vec;

import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

public class MySimpleSentencePreProcessor implements SentencePreProcessor {

    public MySimpleSentencePreProcessor() {
    }

    public String preProcess(String sentence) {
        return (new InputHomogenization(sentence)).transform();
    }
}
