package dl.categorisation;

import dl.categorisation.word2vec.MyWord2Vec;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.stream.Collectors;

/**
 * A partir d'un modèle généré (par word2vec MyWord2Vec)
 * Transformation d'un libellé (texte) en vector.
 */
public class LibelleWord2Vec implements Model {

    private static final Integer VEC_LENGTH = 100;

    private Word2Vec loadedVec = null;

    public LibelleWord2Vec(MyWord2Vec myWord2Vec) {
        this.loadedVec = myWord2Vec.getWord2Vec();
    }

    @Override
    public int getVecLength() {
        return VEC_LENGTH;
    }

    @Override
    public INDArray getVector(String libelle) {
        return getSentence2VecAvg(libelle);
    }

    private INDArray getSentence2VecAvg(String libelle) {

        List<double[]> wordEmbeddings = getWordEmbeddings(libelle);
        INDArray featureVector = Nd4j.zeros(1, VEC_LENGTH);
        for (double[] vector : wordEmbeddings) {
            if (vector != null) {
                INDArray vec = Nd4j.create(vector);
                featureVector = featureVector.add(vec);
            }
        }
        featureVector.divi(wordEmbeddings.size());
        return featureVector;
    }

    public List<double[]> getWordEmbeddings(String libelle) {

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        List<String> tokens = t.create(libelle).getTokens();
        return tokens.stream().map(loadedVec::getWordVector).collect(Collectors.toList());
    }

    public Word2Vec getLoadedVec() {
        return loadedVec;
    }
}
