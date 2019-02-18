package dl.categorisation.iterator;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public abstract class AbstractDataSetIterator implements DataSetIterator {

    protected RecordReader recordReader;
    protected int batchSize = 10;
    protected int batchNum = 0;
    protected DataSet last;
    protected boolean useCurrent = false;
    protected DataSetPreProcessor dataSetPreProcessor;
    protected boolean collectMetaData;
    protected List<String> labels;

    public AbstractDataSetIterator() {}
    public AbstractDataSetIterator(RecordReader recordReader, int batchSize) {
        this.batchSize = batchSize;
        this.recordReader = recordReader;
        this.labels = new ArrayList<>();
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        } else
            return last.numInputs();

    }

    @Override
    public int totalOutcomes() {
        return this.labels != null ? this.labels.size() : 0;
    }

    @Override
    public boolean resetSupported(){
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        batchNum = 0;
        recordReader.reset();
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return recordReader.hasNext();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return this.labels;
    }

    public void setCollectMetaData(boolean collectMetaData) {
        this.collectMetaData = collectMetaData;
    }

    public boolean isCollectMetaData() {
        return collectMetaData;
    }
}
