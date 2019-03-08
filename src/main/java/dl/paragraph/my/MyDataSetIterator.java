package dl.paragraph.my;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Record;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.util.*;

/**
 * Utilisation de la classe CnnSentenceDataSetIterator comme modèle.
 */
public class MyDataSetIterator implements DataSetIterator {

//    public enum UnknownWordHandling {
//        RemoveWord, UseUnknownVector
//    }
//    private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD_SENTINEL";

//    private UnknownWordHandling unknownWordHandling = UnknownWordHandling.RemoveWord;
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private List<String> labels;
    private Map<String, Integer> labelsClassMap;
    private List<Record> records;
    private List<Record> currents;
    private int cursor = 0;
    private final TokenizerFactory tokenizerFactory;

    private boolean avecMontant;
    private boolean avecType;

    public MyDataSetIterator(WordVectors wordVectors, List<Record> records, List<String> labels, int batchSize,
                             int truncateLength, boolean avecMontant, boolean avecType) throws IOException {

        this.wordVectors = wordVectors;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
        this.records = records;
        this.labels = labels;

        // Transformation des labels en integers
        this.labelsClassMap = Maps.newHashMap();
        int idx = 0;
//        Collections.sort(this.labels); // FIXME A voir
        for (String s : this.labels) {
            this.labelsClassMap.put(s, idx++);
        }

        this.batchSize = batchSize;
        this.truncateLength = truncateLength;
        this.avecMontant = avecMontant;
        this.avecType = avecType;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    public List<Record> currents() {
        return currents;
    }

    @Override
    public DataSet next(int num) {

        if (cursor >= records.size()) {
            throw new NoSuchElementException();
        }
        try {
            return nextDataSet(num);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {

        System.out.println(">>> Batch START position : [" + cursor + "]");

        // Tokenize les num prochains libellés
//        List<List<String>> allTokens = Lists.newArrayListWithCapacity(currents.size());
        List<Pair<List<String>, Record>> tokenizedRecord = Lists.newArrayList();
        int maxLength = -1;
        int minLength = Integer.MAX_VALUE;
        currents = Lists.newArrayList();
        Record record;
        for (int i = 0; i < num && cursor < totalExamples(); i++) {
            record = records.get(cursor);
            cursor++; // on avancesur le prochain élément du jeu de données

            List<String> tokens = tokenizeSentence(record.getLibelle());
            if (!tokens.isEmpty()) {
                //Handle edge case: no tokens from sentence
                maxLength = Math.max(maxLength, tokens.size());
                minLength = Math.min(minLength, tokens.size());
                tokenizedRecord.add(Pair.<List<String>, Record>builder().key(tokens).value(record).build());
                currents.add(record);
            } else {
                // Aucun token pour ce libellé, on le saute mais on veut quand même le bon nombre de record dans ce batch
                // donc on rétropédale dans cette boucle mais pas sur le curseur global
                //Skip the current iterator
                System.out.println("## TOKENS vide pour : " + record.getLibelle());
                i--;
            }
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        System.out.println(">>> Batch END position : [" + cursor + "]");
        System.out.println(">>> Batch de [" + currents.size() + "] éléments, maxLength [" + maxLength + "], minLength [" + minLength + "]");

        //Create data for training
        //Here: we have currents.size() examples of varying lengths
        INDArray features = Nd4j.create(new int[]{currents.size(), vectorSize, maxLength}, 'f'); // rank 3
//        INDArray labels = Nd4j.create(new int[]{currents.size(), this.labels.size(), maxLength}, 'f');
        INDArray labels = Nd4j.create(currents.size(), this.labels.size()); // rank 2
//        INDArray labelsMask = Nd4j.zeros(currents.size(), maxLength);


        // Construction du vecteur du libellé : FEATURES
        INDArrayIndex[] idxs = new INDArrayIndex[3];
        idxs[1] = NDArrayIndex.all();
        for (int i = 0; i < tokenizedRecord.size(); i++) {
            idxs[0] = NDArrayIndex.point(i);
            List<String> currSentence = tokenizedRecord.get(i).getKey();
            for (int j = 0; j < currSentence.size() && j < maxLength; j++) {
                idxs[2] = NDArrayIndex.point(j);
                String word = currSentence.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(word);
//                System.out.println(">>> >>> Vector [" + word + "] [" + Arrays.toString(vector.shape()) + "] [" + vector.columns() + "] [" + vector.rows() + "]");
//                INDArray vectorNormi = wordVectors.getWordVectorMatrixNormalized(word);
                features.put(idxs, vector);
            }
//            System.out.println(">>> >>> Features [" + currSentence + "] [" + Arrays.toString(features.shape()) + "]");
        }



        // Get the truncated sequence length of document (i)
//        int seqLength = Math.min(tokens.size(), maxLength);
        // Get all wordvectors for the current document and transpose them to fit the 2nd and 3rd feature shape
//        Collection<String> word = tokens.subList(0, seqLength);
//        final INDArray vectors = wordVectors.getWordVectors(word).transpose();
        // Put wordvectors into features array at the following indices:
        // 1) Document (i)
        // 2) All vector elements which is equal to NDArrayIndex.interval(0, vectorSize)
        // 3) All elements between 0 and the length of the current sequence
//        features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, seqLength)}, vectors);


        // FEATURES MASK
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = null;
        if (minLength != maxLength) {
            featuresMask = Nd4j.zeros(currents.size(), maxLength);
            // Assign "1" to each position where a feature is present, that is, in the interval of [0, seqLength)
//            featuresMask.get(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)}).assign(1);
            for (int i = 0; i < tokenizedRecord.size(); i++) {
                int libelleLength = tokenizedRecord.get(i).getKey().size();
                if (libelleLength >= maxLength) {
                    featuresMask.getRow(i).assign(1.0);
                } else {
                    featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.interval(0, libelleLength)).assign(1.0);
                }
            }
        }


        // Construction du vecteur des LABELS (catégorie)
        for (int i = 0; i < tokenizedRecord.size(); i++) {
            record = tokenizedRecord.get(i).getValue();
            String labelStr = record.getCategorie();
            if (!this.labelsClassMap.containsKey(labelStr)) {
                throw new IllegalStateException("Got label \"" + labelStr + "\" that is not present in list of LabeledSentenceProvider labels");
            }
            int labelIdx = this.labelsClassMap.get(labelStr);
            labels.putScalar(i, labelIdx, 1.0);
        }

        // FIXME A vérifier
//            int lastIdx = Math.min(tokens.size(), maxLength);
//            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);   //Specify that an output exists at the final time step for this example







//        return new DataSet(features, labels, featuresMask, labelsMask);
        return new DataSet(features, labels, featuresMask, (INDArray) null);
    }

    @Override
    public int totalExamples() {
        return records.size();
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return labels.size();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

    public Map<String, Integer> getLabelsClassMap() {
        return Maps.newHashMap(this.labelsClassMap);
    }

    @Override
    public boolean hasNext() {
        return cursor < totalExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }


//    @Override
//    public String nextSentence() {
//
//        if (!hasNext()) {
//            return null;
//        }
//        List<Writable> next = recordReader.next();
//
//        String libelle = next.get(1).toString();
//        String montant = next.get(2).toString();
//        String type = next.get(4).toString();
//
//        if (avecMontant && avecType) {
//            return libelle + " " + type + " " + montant;
//        } else if (avecMontant) {
//            return libelle + " " + montant;
//        } else if (avecType) {
//            return libelle + " " + type;
//        }
//        return libelle;
//    }
//
//    @Override
//    public boolean hasNext() {
//        return recordReader.hasNext();
//    }
//
//    @Override
//    public void reset() {
//        recordReader.reset();
//    }
//
//    @Override
//    public void finish() {
//    }
//
//    @Override
//    public SentencePreProcessor getPreProcessor() {
//        return this.sentencePreProcessor;
//    }
//
//    @Override
//    public void setPreProcessor(SentencePreProcessor sentencePreProcessor) {
//        this.sentencePreProcessor = sentencePreProcessor;
//    }

    /**
     * Used post training to convert a String to a features INDArray that can be passed to the network output method
     *
     * @param reviewContents Contents of the review to vectorize
     * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
     * @return Features array for the given input String
     */
    public INDArray loadFeaturesFromString(String reviewContents, int maxLength){

        List<String> tokens = tokenizeSentence(reviewContents);
        int outputLength = Math.max(maxLength, tokens.size());
        INDArray features = Nd4j.create(1, vectorSize, outputLength);

        int count = 0;
        for (int j = 0; j < tokens.size() && count < maxLength; j++) {
            String token = tokens.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            if (vector == null) {
                continue;   //Word not in word vectors
            }
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
            count++;
        }

        return features;
    }

    private List<String> tokenizeSentence(String sentence) {

        Tokenizer t = tokenizerFactory.create(sentence);
        List<String> tokens = new ArrayList<>();
        while (t.hasMoreTokens()) {
            String token = t.nextToken();
            if (!wordVectors.hasWord(token)) {
//                switch (unknownWordHandling) {
//                    case RemoveWord:
                        continue;
//                    case UseUnknownVector:
//                        token = UNKNOWN_WORD_SENTINEL;
//                }
            }
            tokens.add(token);
        }
        return tokens;
    }
}
