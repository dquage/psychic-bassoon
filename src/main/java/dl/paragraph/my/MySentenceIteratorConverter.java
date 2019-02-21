package dl.paragraph.my;

import dl.paragraph.pojo.Record;
import lombok.NonNull;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.List;

public class MySentenceIteratorConverter implements LabelAwareIterator {
    private MySentenceFromListIterator backendIterator;
    private LabelsSource generator;
    protected static final Logger log = LoggerFactory.getLogger(SentenceIteratorConverter.class);

    public MySentenceIteratorConverter(@NonNull MySentenceFromListIterator iterator) {
        if (iterator == null) {
            throw new NullPointerException("iterator");
        } else {
            this.backendIterator = iterator;
            this.generator = new LabelsSource();
        }
    }

    public boolean hasNextDocument() {
        return this.backendIterator.hasNext();
    }

    public LabelledDocument nextDocument() {
        LabelledDocument document = new LabelledDocument();
        document.setContent(this.backendIterator.nextSentence());
        if (this.backendIterator instanceof LabelAwareSentenceIterator) {
            List<String> labels = ((LabelAwareSentenceIterator)this.backendIterator).currentLabels();
            if (labels != null) {
                Iterator var3 = labels.iterator();

                while(var3.hasNext()) {
                    String label = (String)var3.next();
                    document.addLabel(label);
                    this.generator.storeLabel(label);
                }
            } else {
                String label = ((LabelAwareSentenceIterator)this.backendIterator).currentLabel();
                if (label != null) {
                    document.addLabel(label);
                    this.generator.storeLabel(label);
                }
            }
        } else if (this.generator != null) {
            document.addLabel(this.generator.nextLabel());
        }

        return document;
    }

    public Record record() {
        return backendIterator.current();
    }

    public void reset() {
        this.generator.reset();
        this.backendIterator.reset();
    }

    public boolean hasNext() {
        return this.hasNextDocument();
    }

    public LabelledDocument next() {
        return this.nextDocument();
    }

    public void remove() {
    }

    public LabelsSource getLabelsSource() {
        return this.generator;
    }

    public void shutdown() {
    }
}
