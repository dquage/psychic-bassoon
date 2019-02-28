package dl.paragraph.my;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.util.List;

public class MySentenceIterator implements SentenceIterator {

    private RecordReader recordReader;
    private SentencePreProcessor sentencePreProcessor;
    private boolean avecMontant;

    public MySentenceIterator(RecordReader recordReader, boolean avecMontant) {
        this.recordReader = recordReader;
        this.avecMontant = avecMontant;
    }

    @Override
    public String nextSentence() {

        if (!hasNext()) {
            return null;
        }
        List<Writable> next = recordReader.next();
        if (avecMontant) {
            return next.get(1).toString() + " " + next.get(2).toString();
        }
        return next.get(1).toString();
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
