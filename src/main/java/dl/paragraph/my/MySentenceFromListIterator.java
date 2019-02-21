package dl.paragraph.my;

import dl.paragraph.pojo.Record;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.util.Iterator;
import java.util.List;

public class MySentenceFromListIterator implements SentenceIterator {

    private List<Record> records;
    private Record current;
    private SentencePreProcessor sentencePreProcessor;
    private Iterator<Record> iterator;

    public MySentenceFromListIterator(List<Record> records) {
        this.records = records;
        this.iterator = records.iterator();
    }

    @Override
    public String nextSentence() {

        if (!hasNext()) {
            return null;
        }
        current = iterator.next();
        return current.getLibelle();
    }

    public Record current() {
        return current;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public void reset() {
        iterator = records.iterator();
        current = null;
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
