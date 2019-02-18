package dl;

import dl.categorisation.CategorisationTrain;
import dl.categorisation.LibelleWord2Vec;
import dl.categorisation.word2vec.MyWord2Vec;

/**
 * Test du deep learning avec DL4J
 * https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart
 */
public class Main {

    public static void main(String[] args) {

        try {
            // Création ou chargement du modèle de vectorisation du texte
            MyWord2Vec myWord2Vec = new MyWord2Vec();
            // Classe utilisant le modèle pour transformer du texte en vectors
            LibelleWord2Vec word2Vec = new LibelleWord2Vec(myWord2Vec);
            // Classe de définition, entrainement et évaluation
            CategorisationTrain categorisationTrain = new CategorisationTrain(word2Vec);
            categorisationTrain.train();
            categorisationTrain.evaluate();

        } catch (Exception e) {
            System.out.println("****************Example finished WITH ERROR ********************");
            e.printStackTrace();
        }

        System.out.println("****************Example finished********************");
    }






    public static void main_autre(String[] args) {

        try {
            //////////////////////////////////////////////////////////////////////

            //////////////////////////////////////////////////////////////////////
//            BagOfWords bag = new BagOfWords();
//            bag.train();
            //////////////////////////////////////////////////////////////////////

            //////////////////////////////////////////////////////////////////////
//        CnnSentenceClassificationExample cnnEx = new CnnSentenceClassificationExample();
//        cnnEx.train();
            //////////////////////////////////////////////////////////////////////

//        LibelleVectorsClassifierExample app = new LibelleVectorsClassifierExample();
//        app.makeParagraphVectors();
//        app.checkUnlabeledData();


//        WekaNaivesBayesExemple.process();

//        BasicCSVClassifier.test("depenses2017.data");

//        BasicExemple.test();

//        IrisExemple.test(filenameData);

//        parseDataTest();

//        if (true) return;


        } catch (Exception e) {
            System.out.println("****************Example finished WITH ERROR ********************");
            e.printStackTrace();
        }

        System.out.println("****************Example finished********************");
    }
}
