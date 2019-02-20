package dl;

import dl.paragraph.Categorisation;

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
            categorisation.train();
            categorisation.evaluate();
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
