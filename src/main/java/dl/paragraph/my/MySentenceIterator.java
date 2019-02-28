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
    private boolean avecType;

    public MySentenceIterator(RecordReader recordReader, boolean avecMontant, boolean avecType) {
        this.recordReader = recordReader;
        this.avecMontant = avecMontant;
        this.avecType = avecType;
    }

    @Override
    public String nextSentence() {

        if (!hasNext()) {
            return null;
        }
        List<Writable> next = recordReader.next();

        String libelle = next.get(1).toString();
        String montant = next.get(2).toString();
        String type = next.get(4).toString();

        if (avecMontant && avecType) {
            return libelle + " " + type + " " + montant;
        } else if (avecMontant) {
            return libelle + " " + montant;
        } else if (avecType) {
            return libelle + " " + type;
        }
        return libelle;
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
