package dl.paragraph;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.util.List;

public class MySentenceIterator implements SentenceIterator {

    private RecordReader recordReader;
    private SentencePreProcessor sentencePreProcessor;

    public MySentenceIterator(RecordReader recordReader) {
        this.recordReader = recordReader;
    }

    @Override
    public String nextSentence() {

        if (!hasNext()) {
            return null;
        }
        List<Writable> next = recordReader.next();
        Writable writable = next.get(0);
        return writable != null ? writable.toString() : null;
    }

    @Override
    public boolean hasNext() {
        return recordReader.hasNext();
    }

    @Override
    public void reset() {
        recordReader.reset();
    }

    @Override
    public void finish() {
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return this.sentencePreProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor sentencePreProcessor) {
        this.sentencePreProcessor = sentencePreProcessor;
    }
}
