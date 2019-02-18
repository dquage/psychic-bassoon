package dl.categorisation;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Model {
    int getVecLength();
    INDArray getVector(String libelle);
}
