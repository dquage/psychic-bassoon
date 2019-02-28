package dl.paragraph;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Record;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformProcessRecordReader;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class CSVRetouches {

    private ParagraphVectors paragraphVectors;
    private boolean avecMontant = false;
    private List<String> categoriesLoaded = null;

    public CSVRetouches() {
    }

    public static LocalTransformProcessRecordReader get(String csvFileClasspath) throws IOException, InterruptedException {
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Id")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "Categorie", "Compte", "Type")
                .build();

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Id")
                .build();
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);
        return transformProcessRecordReader;
    }

    public static LocalTransformProcessRecordReader getNormalise(String csvFileClasspath) throws IOException, InterruptedException {
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Categorie", "Libelle")
                .addColumnDouble("Montant")
                .addColumnsString("Compte", "Type")
                .build();

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .build();
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);
        return transformProcessRecordReader;
    }

    public static List<String> readAllCategories(String csvFileClasspath) throws IOException, InterruptedException {

        LocalTransformProcessRecordReader transformProcessRecordReader = get(csvFileClasspath);
        List<String> categories = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            String cat = transformProcessRecordReader.next().get(2).toString();
            if (!cat.isEmpty() && !categories.contains(cat)) {
                categories.add(cat);
            }
        }

        System.out.println("");
        System.out.println("");
        System.out.println("");
        for (String category : categories) {
            Apollon apollon = Apollon.fromLabel(category);
            if (apollon == null) {
                System.out.println("if (\"" + category + "\".equals(cat)) return \"\";");
            }
        }
        System.out.println("");
        System.out.println("");
        System.out.println("");
        return categories;
    }

    /**
     * A partir des numéros de comptes change les labels (catégories)
     * pour n'avoir que les transactions avec des catégories connues (prévues).
     *
     * @param csvFileClasspath
     */
    public static void normaliseDataset(String csvFileClasspath) throws Exception {

        LocalTransformProcessRecordReader transformProcessRecordReader = get(csvFileClasspath);

        List<Record> records = Lists.newArrayList();

        int nbLignesRetirees = 0;
        List<String> lignes = Lists.newArrayList();
        List<String> categoriesRetirees = Lists.newArrayList();
        Map<String, Integer> categorieCount = Maps.newHashMap();

        // 1ere passe
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            String cat = next.get(2).toString();
            String libelle = next.get(1).toString();
            double montant = next.get(0).toDouble();
            String compte = next.get(3).toString();
            String type = next.get(4).toString();

            Apollon apollon = Apollon.fromNumero(compte);
            if (apollon == null) {
                // 2eme check avec label cette fois
                // On traite les cas de numéro racine connus (ex. 455XXXX, 164XXXX)
                cat = adapteCategorie(cat, libelle, compte);
                apollon = Apollon.fromLabel(cat);
            }

            if (apollon == null) {
                if (cat != null && !categoriesRetirees.contains(cat)) {
                    categoriesRetirees.add(cat);
                }
                nbLignesRetirees++;
                continue; // On ne connait pas la catégorie on zap la ligne
            }


            Integer count = categorieCount.get(apollon.getLabel());
            if (count != null) {
                count += 1;
                categorieCount.put(apollon.getLabel(), count);
            } else {
                categorieCount.put(apollon.getLabel(), 1);
            }

            records.add(Record.builder()
                    .type(Record.Type.fromString(type))
                    .apollon(apollon)
                    .categorie(cat)
                    .libelle(libelle)
                    .montant(montant)
                    .numeroCompte(compte)
                    .build());

        }

//        for (String cat : categorieCount.keySet()) {
//            if (categorieCount.get(cat) < 20) {
//                System.out.println("-- " + cat + " : " + categorieCount.get(cat));
//            }
//        }

        // 2 passe + résultat
        for (Record record : records) {

            Apollon apollon = record.getApollon();
            String libelle = record.getLibelle();

            if (apollon == null) {
                continue; // Pas possible normalement
            }

            // On ne prend pas les éléments appartenant à une catégorie très peu représentée
            Integer countCategorie = categorieCount.get(apollon.getLabel());
            if (countCategorie == null || countCategorie < 5) {
                System.out.println("-- Retrait de la catégorie " + apollon.getLabel() + " car elle n'apparait que " + countCategorie + " fois");
                if (apollon.getLabel() != null && !categoriesRetirees.contains(apollon.getLabel())) {
                    categoriesRetirees.add(apollon.getLabel());
                }
                nbLignesRetirees++;
                continue;
            }

            // Normalise libellé
            libelle = normaliseLibelle(libelle);

            // Règles de retrait (ex. chèques)
            if (aRetirer(apollon, libelle)) {
                nbLignesRetirees++;
                continue;
            }

            String ligne = "\"" + record.getApollon().getLabel() + "\",\"" + libelle + "\",\"" + String.valueOf(record.getMontant())
                    + "\",\"" + record.getNumeroCompte() + "\",\"" + record.getType() + "\"";
            lignes.add(ligne);
        }

        System.out.println("Traitement - retrait de " + nbLignesRetirees + " lignes");
        for (String categoriesRetiree : categoriesRetirees) {
            System.out.println("CATEG retirée : " + categoriesRetiree);
        }

        // Enregistrement
        String outputPath = "apollon_data_2018.normalise.csv";
        FileWriter fw = new FileWriter(outputPath);
        for (int i = 0; i < lignes.size(); i++) {
            if (i < lignes.size()-1) {
                fw.write(lignes.get(i) + "\n");
            } else {
                fw.write(lignes.get(i));
            }
        }
        fw.close();
    }

    private static String normaliseLibelle(String libelle) {

        if (libelle == null) {
            return null;
        }

        return libelle.trim().replace(" _", "").replace(" .", "").toUpperCase();
    }

    /**
     * Création d'un jeu de données d'entrainement.
     * Prend aléatoirement le percentage % de données (lignes) du fichier renseigné.
     *
     * FAIRE un normalise avant.
     *
     * @param csvFileClasspath
     */
    public static void createTrainDataset(String csvFileClasspath, int percentage) throws Exception {

        LocalTransformProcessRecordReader transformProcessRecordReader = getNormalise(csvFileClasspath);

        List<String> lignes = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            String cat = next.get(0).toString();
            String libelle = next.get(1).toString();
            String montant = next.get(2).toString();
            String compte = next.get(3).toString();
            String type = next.get(4).toString();
            String ligne = "\"" + cat + "\",\"" + libelle + "\",\"" + montant + "\",\"" + compte + "\",\"" + type + "\"";
            lignes.add(ligne);
        }

        int nbItems = lignes.size();
        double nbItemsForTain = nbItems * (percentage / 100.0);
        System.out.println("Train avec " + nbItemsForTain + " éléments");

        String outputPathTrain = "apollon_data_2018.train.csv";
        FileWriter fwTrain = new FileWriter(outputPathTrain);
        Random rand = new Random();
        for (int i = 0; i < nbItemsForTain; i++) {
            int randomIndex = rand.nextInt(lignes.size());
            String randomElement = lignes.get(randomIndex);
            lignes.remove(randomIndex);
            fwTrain.write(randomElement + "\n");
        }
        fwTrain.close();

        // Ce qui reste est le jeu de test
        String outputPathTest = "apollon_data_2018.test.csv";
        FileWriter fwTest = new FileWriter(outputPathTest);
        for (String ligne : lignes) {
            fwTest.write(ligne + "\n");
        }
        fwTest.close();

    }

    private static boolean aRetirer(Apollon cat, String libelle) {
        return (libelle.contains("CHEQUE") && !libelle.contains("VIR"))
                || (cat == Apollon.COMPTE_ATTENTE)
                || (cat == Apollon.COMPTE_ATTENTE_IMMOBILISATIONS);
    }

    private static String adapteCategorie(String cat, String libelle, String compte) {

        if (compte != null) {
            if (compte.startsWith("164")) {
                return "EMPRUNT";
            }
            if (compte.startsWith("455")) {
                return "COMPTE ASSOCIE";
            }
            if (compte.equals("6464000")) {
                return "COTISATIONS COMPLEMENTAIRES EXPLOITANT";
            }
            if (compte.equals("1080980")) {
                return "CARBURANT VEHICULE";
            }
            if (compte.equals("1080970")) {
                return "ASSURANCE VEHICULE";
            }
            if (compte.equals("4720000")) {
                return "COMPTE DATTENTE DES IMMOBILISATIONS";
            }
        }



//        CATEG retirée : COTISATIONS URSSAF DE L'EXPLOITANT
//        CATEG retirée : COMPTE DE L'EXPLOITANT
//        CATEG retirée : RETRAITE MUTEX
//        CATEG retirée : RETRAITE GAN
//        CATEG retirée : URSSAF
//        CATEG retirée : ACHATS A USAGE UNIQUE
//        CATEG retirée : COTISATION MALADIE EXPLOITANT
//        CATEG retirée : GENERALI - CT 563802057
//        CATEG retirée : LOCATION LD AUTRE QUE VÉHICULE
//        CATEG retirée : CAISSE
//        CATEG retirée : PRODUITS ENTRETIEN
//        CATEG retirée : CONSOMABLE F. BUREAU
//        CATEG retirée : RETRAITE AXA
//        CATEG retirée : PETIT MATERIEL REUTILISABLE
//        CATEG retirée : DEPENSES MIXTES
//        CATEG retirée : FRAIS D'ACTES ET DE CONTENTIEUX
//        CATEG retirée : LEASING AUTRE QUE VÉHICULE
//        CATEG retirée : ACHATS MARCHANDISE
//        CATEG retirée : LOCATION AUTRE QUE VÉHICULE



//        CATEG retirée : RÉINTÉGRATION COMPTABLE
//        CATEG retirée : IMMO.MATERIELS DE BUREAU ET INFORMATIQUE
//        CATEG retirée : AGIPI - RETRAITE COMPLEMENTAIRE
//        CATEG retirée : COMPTE D'ATTENTE DES IMMOBILISATIONS
//        CATEG retirée : REPAS DU MIDI
//        CATEG retirée : AGIOS
//        CATEG retirée : COTISATIONS PROFESSIONNELLES
//        CATEG retirée : HONORAIRES RÉTROCÉDÉS JEREMY
//        CATEG retirée : HONORAIRES RÉTROCÉDÉS SANDRA
//        CATEG retirée : HONORAIRES RÉTROCÉDÉS SYLVIA
//        CATEG retirée : RETRAITE MACSF
//        CATEG retirée : ENTRETIEN ET REPARATIONS cab inf.
//                CATEG retirée : COMPTE D'ATTENTE
//        CATEG retirée : LEASING mini

        if ("COTISATIONS URSSAF DE L'EXPLOITANT".equals(cat)) return "COTISATIONS URSSAF DE L EXPLOITANT";
        if ("COMPTE DE L'EXPLOITANT".equals(cat)) return "COMPTE DE L EXPLOITANT";
        if ("RETRAITE MUTEX".equals(cat)) return "COTISATIONS COMPLEMENTAIRES RETRAITE";
        if ("RETRAITE GAN".equals(cat)) return "COTISATIONS COMPLEMENTAIRES RETRAITE";
        if ("URSSAF".equals(cat)) return "COTISATIONS URSSAF DE L EXPLOITANT";
        if ("ACHATS A USAGE UNIQUE".equals(cat)) return "ACHATS USAGE UNIQUE";
        if ("COTISATION MALADIE EXPLOITANT".equals(cat)) return "COTISATIONS MALADIE EXPLOITANT";
        if ("COMPTE ASSOCIES".equals(cat)) return "COMPTE ASSOCIE";
        if ("COMPTE ASSOCIE N°1".equals(cat)) return "COMPTE ASSOCIE";
        if ("COMPTE ASSOCIE N°2".equals(cat)) return "COMPTE ASSOCIE";
        if ("COMPTE ASSOCIE N°3".equals(cat)) return "COMPTE ASSOCIE";
        if ("COMPTE ASSOCIE N°4".equals(cat)) return "COMPTE ASSOCIE";
        if ("COMPTE ASSOCIE N°5".equals(cat)) return "COMPTE ASSOCIE";
        if ("CAISSE".equals(cat)) return ""; // Prend pas en compte

        if ("GENERALI - CT 563802057".equals(cat)) return "COTISATIONS COMPLEMENTAIRES RETRAITE";
        if ("LOCATION LD AUTRE QUE VÉHICULE".equals(cat)) return "LOCATION LD AUTRE QUE VEHICULE";

        if ("PRODUITS ENTRETIEN".equals(cat)) return "PRODUITS D ENTRETIEN";
        if ("CONSOMABLE F. BUREAU".equals(cat)) return "FOURNITURES DE BUREAU ET ADMINISTRATIVES";
        if ("COMPLEMENTAIRE FRAIS DE SANTE".equals(cat)) return "COTISATIONS COMPLEMENTAIRES EXPLOITANT";
        if ("RETRAITE AXA".equals(cat)) return "COTISATIONS COMPLEMENTAIRES RETRAITE";
        if ("PETIT MATERIEL REUTILISABLE".equals(cat)) return "PETIT OUTILLAGE PETIT MATERIEL";
        if ("DEPENSES MIXTES".equals(cat)) return "PART PRIVEE DES DEPENSES MIXTES";
        if ("FRAIS D'ACTES ET DE CONTENTIEUX".equals(cat)) return "FRAIS D ACTES ET DE CONTENTIEUX";
        if ("LEASING AUTRE QUE VÉHICULE".equals(cat)) return "LEASING AUTRE QUE VEHICULE";
        if ("ACHATS MARCHANDISE".equals(cat)) return "ACHATS MARCHANDISES";
        if ("LOCATION AUTRE QUE VÉHICULE".equals(cat)) return "LOCATION LD AUTRE QUE VEHICULE";
        if ("RÉINTÉGRATION COMPTABLE".equals(cat)) return "REINTEGRATION COMPTABLE";
//        if ("IMMO.MATERIELS DE BUREAU ET INFORMATIQUE".equals(cat)) return "";
//        if ("COTISATIONS MADELIN".equals(cat)) return "";
//        if ("COMPTE ASSOCIES".equals(cat)) return "";
//        if ("AGIPI - RETRAITE COMPLEMENTAIRE".equals(cat)) return "";
//        if ("COMPTE D'ATTENTE DES IMMOBILISATIONS".equals(cat)) return "";
//        if ("COMPTE ASSOCIE N°4".equals(cat)) return "";
//        if ("REPAS DU MIDI".equals(cat)) return "";
//        if ("AGIOS".equals(cat)) return "";
//        if ("COTISATIONS PROFESSIONNELLES".equals(cat)) return "";
//        if ("HONORAIRES RÉTROCÉDÉS JEREMY".equals(cat)) return "";
//        if ("HONORAIRES RÉTROCÉDÉS SANDRA".equals(cat)) return "";
//        if ("HONORAIRES RÉTROCÉDÉS SYLVIA".equals(cat)) return "";
//        if ("RETRAITE MACSF".equals(cat)) return "";
//        if ("ENTRETIEN ET REPARATIONS cab inf.".equals(cat)) return "";
//        if ("COMPTE D'ATTENTE".equals(cat)) return "";
//        if ("LEASING mini".equals(cat)) return "";


        return cat;
    }
}
