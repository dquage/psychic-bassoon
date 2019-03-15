package dl;

import dl.paragraph.CSVRetouches;

import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test du deep learning avec DL4J
 * https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart
 */
public class MainRetouches {

    public static void main(String[] args) {

        setJavaLogger(Level.INFO);
        try {

//            CSVRetouches.verifyDataset("apollon_data_2018.normalise.csv");

            // Permet de normaliser le jeux complet de données (vire des libellés non concluants, normalise un peu les libellés...)
            // Sépare le jeux de données complet en 1 jeu de train, 1 jeu de test à XX%
            CSVRetouches.normaliseDataset("donnees/apollon_data_2018_final.csv");
            CSVRetouches.createTrainDataset("apollon_data_2018.normalise.csv", 90);

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
