package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import dl.paragraph.acceptance.AcceptanceComparative;
import dl.paragraph.acceptance.AcceptanceType;
import dl.paragraph.pojo.Categorie;
import dl.paragraph.pojo.Resultat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.joda.time.DateTime;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class Tracing {

    public static void saveResultats(List<Resultat> resultats) throws IOException {

        // Stats accuracy
        int nbItems = resultats.size();
        int nbItemsAccurates = 0;
        int nbItemsAcceptedAccurates = 0;
        int nbItemsAcceptedInaccurates = 0;
        double scores1Sum = 0;
        double scores1AccurateSum = 0;

        // Calculs de la justesse des prédictions par tranches de précision
        double nbItemsPrecision80minSum = 0;
        double nbItemsPrecision70minSum = 0;
        double nbItemsPrecision60minSum = 0;
        double nbItemsPrecision50minSum = 0;
        double nbItemsPrecision40minSum = 0;
        double nbItemsPrecision30minSum = 0;
        double nbItemsAccuratePrecision80minSum = 0;
        double nbItemsAccuratePrecision70minSum = 0;
        double nbItemsAccuratePrecision60minSum = 0;
        double nbItemsAccuratePrecision50minSum = 0;
        double nbItemsAccuratePrecision40minSum = 0;
        double nbItemsAccuratePrecision30minSum = 0;

        for (Resultat resultat : resultats) {
            double score = resultat.getCategorie1().getScore() * 100;
            scores1Sum += score;

            if (score > 80) {
                nbItemsPrecision80minSum++;
            } else if (score > 70) {
                nbItemsPrecision70minSum++;
            } else if (score > 60) {
                nbItemsPrecision60minSum++;
            } else if (score > 50) {
                nbItemsPrecision50minSum++;
            } else if (score > 40) {
                nbItemsPrecision40minSum++;
            } else {
                nbItemsPrecision30minSum++;
            }

            if (resultat.isAccurate()) {
                nbItemsAccurates++;
                scores1AccurateSum += score;

                if (score > 80) {
                    nbItemsAccuratePrecision80minSum++;
                } else if (score > 70) {
                    nbItemsAccuratePrecision70minSum++;
                } else if (score > 60) {
                    nbItemsAccuratePrecision60minSum++;
                } else if (score > 50) {
                    nbItemsAccuratePrecision50minSum++;
                } else if (score > 40) {
                    nbItemsAccuratePrecision40minSum++;
                } else {
                    nbItemsAccuratePrecision30minSum++;
                }
            }

            if (resultat.isAccurateLabelAccepted()) {
                nbItemsAcceptedAccurates++;
            } else {
                if (resultat.getLabelExpected() != null) {
                    nbItemsAcceptedInaccurates++;
                }
            }
        }

        String moyenneScores1All = nbItems > 0 ? String.format("%2.2f", (scores1Sum / nbItems)) : "0";
        String moyenneScores1Accurate = nbItemsAccurates > 0 ? String.format("%2.2f", (scores1AccurateSum / nbItemsAccurates)) : "0";

        String tauxJustesse = nbItems > 0 ? String.format("%2.2f", (((double) nbItemsAccurates * 100) / nbItems)) : "0";
        String tauxJustesseAccepted = nbItems > 0 ? String.format("%2.2f", (((double) nbItemsAcceptedAccurates * 100) / nbItems)) : "0";
        String tauxErreurAccepted = nbItems > 0 ? String.format("%2.2f", (((double) nbItemsAcceptedInaccurates * 100) / nbItems)) : "0";
        String tauxJustesse80min = nbItemsPrecision80minSum > 0 ? String.format("%2.2f", (nbItemsAccuratePrecision80minSum * 100 / nbItemsPrecision80minSum)) : "0";
        String tauxJustesse70min = nbItemsPrecision70minSum > 0 ? String.format("%2.2f", (nbItemsAccuratePrecision70minSum * 100 / nbItemsPrecision70minSum)) : "0";
        String tauxJustesse60min = nbItemsPrecision60minSum > 0 ? String.format("%2.2f", (nbItemsAccuratePrecision60minSum * 100 / nbItemsPrecision60minSum)) : "0";
        String tauxJustesse50min = nbItemsPrecision50minSum > 0 ? String.format("%2.2f", (nbItemsAccuratePrecision50minSum * 100 / nbItemsPrecision50minSum)) : "0";
        String tauxJustesse40min = nbItemsPrecision40minSum > 0 ? String.format("%2.2f", (nbItemsAccuratePrecision40minSum * 100 / nbItemsPrecision40minSum)) : "0";
        String tauxJustesse30min = nbItemsPrecision30minSum > 0 ? String.format("%2.2f", (nbItemsAccuratePrecision30minSum * 100 / nbItemsPrecision30minSum)) : "0";

        System.out.println("Nombre éléments catégorisés avec succès : " + nbItemsAccurates + " donc " + tauxJustesse + "%");
        System.out.println("Nombre éléments acceptés catégorisés avec SUCCES : " + nbItemsAcceptedAccurates + " donc " + tauxJustesseAccepted + "%");
        System.out.println("Nombre éléments acceptés catégorisés en   ERREUR : " + nbItemsAcceptedInaccurates + " donc " + tauxErreurAccepted + "%");


        StringBuilder sb = new StringBuilder();

        sb.append("Résultats : ");
        sb.append("\n");
        sb.append("Nombre éléments testés : ").append(nbItems);
        sb.append("\n");
        sb.append("Nombre éléments catégorisés avec succès : ").append(nbItemsAccurates).append(" donc ").append(tauxJustesse).append("%");
        sb.append("\n");
        sb.append("Nombre éléments acceptés catégorisés avec succès : ").append(nbItemsAcceptedAccurates).append(" donc ").append(tauxJustesseAccepted).append("%");
        sb.append("\n");
        sb.append("Nombre éléments acceptés catégorisés avec erreur : ").append(nbItemsAcceptedInaccurates).append(" donc ").append(tauxErreurAccepted).append("%");
        sb.append("\n");
        sb.append("\n");
        sb.append("Moyenne de la précision de la meilleure catégorie trouvée : ").append(moyenneScores1All).append("%");
        sb.append("\n");
        sb.append("Moyenne de la précision des catégories exactes trouvées : ").append(moyenneScores1Accurate).append("%");

        sb.append("\n");
        sb.append("Précision > 80% : ").append(nbItemsPrecision80minSum).append(" dont ")
                .append(nbItemsAccuratePrecision80minSum).append(" justes donc ").append(tauxJustesse80min).append("%");
        sb.append("\n");
        sb.append("Précision > 70% : ").append(nbItemsPrecision70minSum).append(" dont ")
                .append(nbItemsAccuratePrecision70minSum).append(" justes donc ").append(tauxJustesse70min).append("%");
        sb.append("\n");
        sb.append("Précision > 60% : ").append(nbItemsPrecision60minSum).append(" dont ")
                .append(nbItemsAccuratePrecision60minSum).append(" justes donc ").append(tauxJustesse60min).append("%");
        sb.append("\n");
        sb.append("Précision > 50% : ").append(nbItemsPrecision50minSum).append(" dont ")
                .append(nbItemsAccuratePrecision50minSum).append(" justes donc ").append(tauxJustesse50min).append("%");
        sb.append("\n");
        sb.append("Précision > 40% : ").append(nbItemsPrecision40minSum).append(" dont ")
                .append(nbItemsAccuratePrecision40minSum).append(" justes donc ").append(tauxJustesse40min).append("%");
        sb.append("\n");
        sb.append("Précision > 30% : ").append(nbItemsPrecision30minSum).append(" dont ")
                .append(nbItemsAccuratePrecision30minSum).append(" justes donc ").append(tauxJustesse30min).append("%");

        sb.append("\n");
        sb.append("\n");
        sb.append("\n");

        for (Resultat resultat : resultats) {
            if (resultat.getRecord() != null) {
                sb.append(resultat.getRecord().isRecette() ? "[RECETTE]" : "[DEPENSE]");
                sb.append(" ").append(resultat.getRecord().getMontant()).append("€ | ");
            }
            sb.append(resultat.getLibelle());
            if (resultat.getLabelAccepted() != null) {
                sb.append("\n");
                sb.append("[RETENU][").append(resultat.getLabelAccepted()).append("]");
                if (resultat.getLabelExpected() != null && !resultat.isAccurateLabelAccepted()) {
                    sb.append("[ERREUR]");
                }
            }
            if (resultat.getLabelExpected() != null) {
                sb.append("\n");
                sb.append("[EXPC][").append(resultat.getLabelExpected()).append("]");
            }
            sb.append("\n");
            sb.append("[").append(resultat.getCategorie1().getScoreArrondi()).append("][").append(resultat.getCategorie1().getLabel()).append("]");
            sb.append("\t");
            sb.append("[").append(resultat.getCategorie2().getScoreArrondi()).append("][").append(resultat.getCategorie2().getLabel()).append("]");
            sb.append("\t");
            sb.append("[").append(resultat.getCategorie3().getScoreArrondi()).append("][").append(resultat.getCategorie3().getLabel()).append("]");
            sb.append("\n");
            sb.append("\n");
        }

        DateTime date = DateTime.now();
        File file = new File("resultats", "evaluation_" + date.toString("yyyyMMddHHmmss") + ".txt");
        byte[] strToBytes = sb.toString().getBytes();
        Files.write(file.toPath(), strToBytes,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);
    }

    public static void saveResultatsParCategorie(List<Resultat> resultats) throws IOException {


        Map<String, List<String>> categs = Maps.newHashMap();
        for (String category : Categorisation.CATEGORIES) {
            List<String> libelles = Lists.newArrayList();
            categs.put(category, libelles);
            for (Resultat resultat : resultats) {
                if (resultat.getCategorie1().getLabel().equals(category)) {
                    libelles.add(resultat.getLibelle());
                }
            }
        }


        StringBuilder sb = new StringBuilder();
        for (String category : Categorisation.CATEGORIES) {
            sb.append(category).append("\t").append(categs.get(category).size());
            sb.append("\n");
            sb.append("\n");
            for (Resultat resultat : resultats) {
                if (resultat.getCategorie1().getLabel().equals(category)) {
                    sb.append("[").append(resultat.getCategorie1().getScoreArrondi()).append("] ");
                    sb.append(resultat.getLibelle());
                    sb.append("\n");
                }
            }
            sb.append("\n");
        }

        DateTime date = DateTime.now();
        File file = new File("resultats", "evaluation_par_categorie_" + date.toString("yyyyMMddHHmmss") + ".txt");
        byte[] strToBytes = sb.toString().getBytes();
        Files.write(file.toPath(), strToBytes,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING);

    }

}
