package dl.paragraph.my;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
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
public class Word2IdDataSetIterator implements DataSetIterator {

//    public enum UnknownWordHandling {
//        RemoveWord, UseUnknownVector
//    }
//    private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD_SENTINEL";

//    private UnknownWordHandling unknownWordHandling = UnknownWordHandling.RemoveWord;
    private final int batchSize;
    private final int truncateLength;

    private List<String> labels;
    private Map<String, Integer> labelsClassMap;
    private List<Record> records;
    private List<Record> currents;
    private int cursor = 0;
    private final TokenizerFactory tokenizerFactory;
    private Map<String, Integer> word2id;

    private boolean avecMontant;
    private boolean avecType;

    public Word2IdDataSetIterator(List<Record> records, List<String> labels, int batchSize,
                                  int truncateLength, boolean avecMontant, boolean avecType, Map<String, Integer> word2id) throws IOException {
        this.records = records;
        this.labels = labels;

        // Transformation des labels en integers
        this.labelsClassMap = Maps.newHashMap();
        int idx = 0;
        for (String s : this.labels) {
            this.labelsClassMap.put(s, idx++);
        }

        this.batchSize = batchSize;
        this.truncateLength = truncateLength;
        this.avecMontant = avecMontant;
        this.avecType = avecType;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        this.word2id = word2id;

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
            return nextDataSetWid(num);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    private DataSet nextDataSetWid(int num) throws IOException {

        System.out.println(">>> Batch START position : [" + cursor + "] ("+num+")");

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

            if (!this.labelsClassMap.containsKey(record.getCategorie())) {
                i--;
                continue;
            }

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
        INDArray features = Nd4j.create(new int[]{currents.size(), 1, truncateLength}); // rank 3 FIXME: tester virer 'f'
        INDArray labels = Nd4j.create(new int[]{currents.size(), this.labels.size(), truncateLength});
        INDArray featuresMask = Nd4j.zeros(new int[]{currents.size(), truncateLength}); // positions des vecteurs de features pour lesquels la valeur est une donnée (ex. pour un vecteur plus court que maxlength, la fin du vecteur aura des zeros dans featuresMask)
        INDArray labelsMask = Nd4j.zeros(new int[]{currents.size(), truncateLength});


        // Construction du vecteur du libellé : FEATURES
        for (int i = 0; i < tokenizedRecord.size(); i++) {

            List<String> tokens = tokenizedRecord.get(i).getKey();
            int seqLength = Math.min(tokens.size(), truncateLength);

            String labelStr = tokenizedRecord.get(i).getValue().getCategorie();
            if (!this.labelsClassMap.containsKey(labelStr)) {
                throw new IllegalStateException("Got label \"" + labelStr + "\" that is not present in list of LabeledSentenceProvider labels");
            }
            int labelIdx = this.labelsClassMap.get(labelStr);

            for (int j = 0; j < seqLength; j++) {
                int id = this.word2id.get(tokens.get(j));
                features.putScalar(new int[]{i, 0, j}, id);
                featuresMask.putScalar(new int[]{i, j}, 1.0);
            }

            labels.putScalar(new int[]{i, labelIdx, 0}, 1.0);
            labelsMask.putScalar(new int[]{i, 0}, 1.0);

        }

//        return new DataSet(features, labels, featuresMask, (INDArray) null);
        return new DataSet(features, labels, featuresMask, labelsMask);
    }



    private DataSet nextDataSetWidFlat(int num) throws IOException {

        System.out.println(">>> Batch START position : [" + cursor + "] ("+num+")");

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
        INDArray features = Nd4j.create(new int[]{currents.size(), word2id.size()}, 'f'); // rank 3
//        INDArray featuresMask = Nd4j.zeros(currents.size(), truncateLength); // positions des vecteurs de features pour lesquels la valeur est une donnée (ex. pour un vecteur plus court que maxlength, la fin du vecteur aura des zeros dans featuresMask)

        INDArray labels = Nd4j.create(currents.size(), this.labels.size());
//        INDArray labelsMask = Nd4j.zeros(currents.size(), truncateLength);


        // Construction du vecteur du libellé : FEATURES
        for (int i = 0; i < tokenizedRecord.size(); i++) {

            List<String> tokens = tokenizedRecord.get(i).getKey();
            int seqLength = Math.min(tokens.size(), maxLength);

            String labelStr = tokenizedRecord.get(i).getValue().getCategorie();
            if (!this.labelsClassMap.containsKey(labelStr)) {
                throw new IllegalStateException("Got label \"" + labelStr + "\" that is not present in list of LabeledSentenceProvider labels");
            }
            int labelIdx = this.labelsClassMap.get(labelStr);

            for (int j = 0; j < seqLength; j++) {
                int id = this.word2id.get(tokens.get(j));
                features.putScalar(new int[]{i, id}, 1.0+features.getDouble(new int[]{i, id}));
//                featuresMask.putScalar(new int[]{i, j}, 1.0);
           }
            int sum=0;
            for (int j = 0; j < word2id.size(); j++) {
                sum += features.getDouble(new int[]{i, j});
//                System.out.print(features.getDouble(new int[]{i, j})+" ");
            }
            System.out.println(sum+" <=> "+seqLength);

//            labels.putScalar(new int[]{i, 0, seqLength-1}, labelIdx);
//            labelsMask.putScalar(new int[]{i, seqLength-1}, 1.0);

            labels.putScalar(new int[]{i, labelIdx}, 1.0);
//            labelsMask.putScalar(new int[]{i, seqLength-1}, 1.0);

//            System.out.println(">>> >>> Features [" + currSentence + "] [" + Arrays.toString(features.shape()) + "]");
        }

//        return new DataSet(features, labels, featuresMask, (INDArray) null);
        return new DataSet(features, labels);
    }

    @Override
    public int totalExamples() {
        return records.size();
    }

    @Override
    public int inputColumns() {
        return truncateLength;
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



    private List<String> tokenizeSentence(String sentence) {

        Tokenizer t = tokenizerFactory.create(sentence);
        List<String> tokens = new ArrayList<>();
        while (t.hasMoreTokens()) {
            String token = t.nextToken().toLowerCase();
            if (!word2id.containsKey(token))
                token = "<unk>";
            tokens.add(token);
        }
        return tokens;
    }
}
