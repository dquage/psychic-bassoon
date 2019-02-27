package dl;

import com.google.common.collect.Lists;
import dl.paragraph.Categorisation;
import dl.paragraph.pojo.Record;

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
            Categorisation categorisation = new Categorisation();
//            categorisation.train();
//            categorisation.evaluate(Categorisation.readCSVRecordsTest("depenses2017.test"));
//            categorisation.evaluate(Categorisation.readCSVRecordsReels("julien.test"));
            categorisation.evaluate(Lists.newArrayList(Record.builder().montant(22L).libelle("virements").build()));


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
