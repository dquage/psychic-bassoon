package dl;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import dl.paragraph.CSVRetouches;
import dl.paragraph.Categorisation;
import dl.paragraph.CategorisationRNN;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Record;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import javax.naming.directory.Attributes;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test du deep learning avec DL4J
 * https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart
 */
public class MainParagraph {

    public static void main(String[] args) {

        setJavaLogger(Level.INFO);
        try {

            // Permet de normaliser le jeux complet de données (vire des libellés non concluants, normalise un peu les libellés...)
//            CSVRetouches.normaliseDataset("donnees/apollon_data_2018.csv");
            // Sépare le jeux de données complet en 1 jeu de train, 1 jeu de test à 90%
//            CSVRetouches.createTrainDataset("apollon_data_2018.normalise.csv", 90);

//            Categorisation categorisation = new Categorisation();
//            // Entrainement et création du modèle (long)
//            categorisation.train();
//            // Evaluation du jeu de test
//            categorisation.evaluate();

            // Catégorisation RNN
            CategorisationRNN categorisationRNN = new CategorisationRNN();
            categorisationRNN.evaluate();



            // Autre tests
//            categorisation.evaluate(Categorisation.readCSVRecordsReels("julien.test"));
//            categorisation.evaluate(Lists.newArrayList(Record.builder().montant(22L).libelle("OD de balance d'ouverture").build()));
            // Affiche juste la liste de toutes les catégories
//            List<String> categories = CSVRetouches.readAllCategories("apollon_data_2018.csv");


        } catch (Exception e) {
            System.out.println("****************Example finished WITH ERROR ********************");
            e.printStackTrace();
        }
        System.out.println("****************Example finished********************");
    }

    private static void setJavaLogger(Level targetLevel) {
        Logger root = Logger.getLogger("");
        root.setLevel(targetLevel);
        for (Handler handler : root.getHandlers()) {
            handler.setLevel(targetLevel);
        }
        System.out.println("level set: " + targetLevel.getName());
    }
}
