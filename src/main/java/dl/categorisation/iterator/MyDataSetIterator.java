package dl.categorisation.iterator;

import dl.categorisation.CategorisationTrain;
import dl.categorisation.LibelleWord2Vec;
import dl.categorisation.Model;
import dl.categorisation.word2vec.MySimpleSentencePreProcessor;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class MyDataSetIterator extends AbstractDataSetIterator {

    private int labelIndex = -1;
    private int featureIndex = -1;
    private MySimpleSentencePreProcessor preProcessor = new MySimpleSentencePreProcessor();
    private Model model;
    private Map<Integer, String> classifier;

    /**
     * Main constructor
     *
     * @param recordReader      the recordreader to use
     * @param batchSize         the batch size
     * @param labelIndex        the index of the label (catégorie)
     * @param featureIndex      the index of the text (libellé)
     */
    public MyDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int featureIndex, Model model) {
        super(recordReader, batchSize);
        this.batchSize = batchSize;
        this.labelIndex = labelIndex;
        this.featureIndex = featureIndex;
        this.model = model;
        this.classifier = CategorisationTrain.readClassifier();
        this.labels = new ArrayList<>(this.classifier.values());
    }

    @Override
    public DataSet next(int num) {
        if (useCurrent) {
            useCurrent = false;
            if (dataSetPreProcessor != null) dataSetPreProcessor.preProcess(last);
            return last;
        }

        List<DataSet> dataSets = new ArrayList<>();
        List<RecordMetaData> meta = (collectMetaData ? new ArrayList<RecordMetaData>() : null);

        for (int i = 0; hasNext() && i < batchSize; i++) {
            if(collectMetaData){
                Record record = recordReader.nextRecord();
                dataSets.add(getDataSet(record.getRecord()));
                meta.add(record.getMetaData());
            } else {
                List<Writable> record = recordReader.next();
                dataSets.add(getDataSet(record));
            }
        }
        batchNum++;

        if(dataSets.isEmpty())
            return new DataSet();

        DataSet ret = DataSet.merge(dataSets);
        if(collectMetaData){
            ret.setExampleMetaData(meta);
        }
        last = ret;
        if (dataSetPreProcessor != null) {
            dataSetPreProcessor.preProcess(ret);
        }
        //Add label name values to dataset
        if (recordReader.getLabels() != null) {
            ret.setLabelNames(recordReader.getLabels());
        }

        return ret;
    }

    private DataSet getDataSet(List<Writable> record) {

        List<Writable> currList;
        if (record != null) {
            currList = record;
        } else {
            currList = new ArrayList<>(record);
        }

        INDArray featureVector = null;
        int labelCount = 0;

        // Label (catégorie)
        final Writable current = currList.get(labelIndex);
        final INDArray label = Nd4j.create(1, 1);
//        label.putScalar(0, 0, current.toInt());

        this.classifier.forEach((key, value) -> {
            if (value != null && value.equals(current.toString())) {
                label.putScalar(0, 0, key);
                return;
            }
        });

        // Feature (Libellés)
        Writable currentFeature = currList.get(featureIndex);
        String value = currentFeature.toString();
        value = preProcessor.preProcess(value);
//        if (model instanceof LibelleWord2Vec) {
//            value = preProcessor.preProcess(value);
//            LibelleWord2Vec libelleWord2Vec = (LibelleWord2Vec) model;
//            if (libelleWord2Vec.getWordEmbeddings(value).stream().noneMatch(Objects::nonNull)) {
//                return new DataSet(Nd4j.zeros(1, model.getVecLength()), Nd4j.zeros(1, 1));
//            }
//        }
        featureVector = model.getVector(value);

        DataSet dataSet = new DataSet(featureVector, label);

        if (dataSet.getFeatures().shape().length > 8) {
            throw new RuntimeException("getDataSet retourne un shape trop grand");
        }

        return dataSet;
    }
}
