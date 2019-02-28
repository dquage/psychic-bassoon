package dl;

import com.google.common.collect.Lists;
import dl.paragraph.CSVRetouches;
import dl.paragraph.Categorisation;
import dl.paragraph.pojo.Record;

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

//            CSVRetouches.normaliseDataset("apollon_data_2018.csv");
//            CSVRetouches.createTrainDataset("apollon_data_2018.normalise.csv", 85);
//            List<String> categories = CSVRetouches.readAllCategories("apollon_data_2018.csv");

            Categorisation categorisation = new Categorisation();
            categorisation.train();
            categorisation.evaluate();

//            categorisation.evaluate(Categorisation.readCSVRecordsReels("julien.test"));
//            categorisation.evaluate(Lists.newArrayList(Record.builder().montant(22L).libelle("OD de balance d'ouverture").build()));


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
