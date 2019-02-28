package dl.paragraph;

import com.google.common.collect.Lists;
import dl.paragraph.pojo.Apollon;
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
import java.util.Random;

public class CSVRetouches {

    public static String[] CATEGORIES = {
            "SERVICES_BANCAIRES", "COMPTE_DE_L_EXPLOITANT", "DEPLACEMENTS_MISSIONS_RECEPTIONS", "REDEVANCE_COLLABORATION", "ESSENCE_VEHICULE",
            "CHARGES_DIVERSES_DE_GESTION_COURANTE", "REPAS_DE_MIDI", "ENTRETIEN_ET_REPARATIONS", "DEPENSES_VEHICULE",
            "EMPRUNT", "COTISATIONS_COMPLEMENTAIRES_EXPLOITANT", "CHARGES_FINANCIERES", "ASSURANCE_VEHICULE", "FRAIS_DE_TELECOMMUNICATION",
            "HONORAIRES_RETROCEDES", "FOURNITURES_DE_BUREAU_ET_ADMINISTRATIVES", "PETIT_OUTILLAGE_PETIT_MATERIEL", "ASSURANCES",
            "COTISATIONS_PROF_ET_SYNDICALE", "FRAIS_POSTAUX", "CFE_CVAE", "ACHATS_A_USAGE_UNIQUE", "TRANSPORTS_DIVERS_BUS_TAXIS",
            "HONORAIRES_RETROCEDES_A_REMPLACANT", "CADEAUX_CLIENTELE", "COTISATION_CARPIMKO", "FRAIS_DE_FORMATION_SEMINAIRE",
            "CHEQUES_VACANCES", "FRAIS_DE_DOCUMENTATION_TECHNIQUE", "LOCATIONS_IMMOBILIERES", "LLD_VEHICULE", "AUTRES_SERVICES_EXTERIEURS_HONORAIRES",
            "HONORAIRES", "AGIOS_BANCAIRES", "PRESTATIONS_DE_SERVICE", "COTISATIONS_MALADIE_EXPLOITANT",
            "COMPTE_ASSOCIE", "AUTRES_IMPOTS", "LEASING_AUTRE_QUE_VEHICULE", "EDF_GDF_CHAUFFAGE", "REMUNERATIONS_DU_PERSONNEL",
            "COTISATIONS_URSSAF_DE_L_EXPLOITANT", "PRODUITS_D_ENTRETIEN", "LEASING_VEHICULE", "COTISATIONS_COMPLEMENTAIRES_SANTE", "PART_PRIVEE_DES_DEPENSES_MIXTES",
            "PERTE_EMPLOI_MADELIN", "FRAIS_DE_CONGRES", "LOCATION_LD_AUTRE_QUE_VEHICULE", "COTISATIONS_COMPLEMENTAIRES_RETRAITE",
            "FRAIS_D_ACTES_ET_DE_CONTENTIEUX", "FOURNISSEURS_D_IMMOBILISATIONS", "ACHATS_MARCHANDISES", "COMPTE_D_ATTENTE", "APPORT_PERSONNEL",
            "SECURITE_SOC_ET_PREVOYANCE_DU_PERSONNEL", "COTISATIONON_CARPIMKO", "CREDIT_BAIL_LEASING_MOBILIER"
    };

    public static String[] CATEGORIES_2018 = {
            "ENTRETIEN ET REPARATIONS", "COTISATION CARPIMKO", "COTISATIONS URSSAF DE L'EXPLOITANT", "COTISATIONS COMPLEMENTAIRES EXPLOITANT",
            "COMPTE DE L'EXPLOITANT", "SERVICES BANCAIRES", "FRAIS DE TELECOMMUNICATION", "AGIOS BANCAIRES", "CARBURANT VEHICULE", "COMPTE DE L EXPLOITANT",
            "RETRAITE MUTEX", "HONORAIRES RETROCEDES", "RETRAITE GAN", "ACHATS USAGE UNIQUE", "ASSURANCES", "FRAIS POSTAUX", "URSSAF", "CADEAUX CLIENTELE",
            "FOURNITURES DE BUREAU ET ADMINISTRATIVES", "CFE,CVAE", "COTISATIONS PROF.ET SYNDICALE", "ASSURANCE VEHICULE", "ACHAT TOYOTA YARIS",
            "CHARGES FINANCIERES", "RACHAT PATIENTÈLE", "COTISATION MALADIE EXPLOITANT", "LOCATIONS IMMOBILIERES", "DEPENSES VEHICULE",
            "PART PRIVEE DES DEPENSES MIXTES", "ACHAT PATIENTÈLE", "PETIT OUTILLAGE PETIT MATERIEL", "DEPLACEMENTS, MISSIONS, RECEPTIONS",
            "PRODUITS D ENTRETIEN", "PRESTATIONS DE SERVICE", "LLD VEHICULE", "TRANSPORTS DIVERS (bus, taxis)", "E.D.F., G.D.F., CHAUFFAGE",
            "FOURNISSEURS D IMMOBILISATIONS", "ACHATS A USAGE UNIQUE", "", "HONORAIRES", "COMPTE ASSOCIE", "COTISATIONS COMPLEMENTAIRES SANTE",
            "COTISATIONS URSSAF DE L EXPLOITANT", "VOITURE PEUGEOT", "PATIENTELE", "CHARGES DIVERSES DE GESTION COURANTE",
            "SECURITE SOC. ET PREVOYANCE DU PERSONNEL", "REMUNERATIONS DU PERSONNEL", "AUTRES SERVICES EXTERIEURS : HONORAIRES", "LEASING VEHICULE",
            "EMPRUNT PATIENTELE", "COTISATIONS MALADIE EXPLOITANT", "FRAIS DE DOCUMENTATION TECHNIQUE", "REMBOURSEMENT INDUS", "VOLVO V40",
            "FRAIS D ACTES ET DE CONTENTIEUX", "ACHAT APPLE 2018", "COMPTE ASSOCIE N°2", "AUTRES IMPOTS", "EMPRUNT ENVELOPPE PRO", "ACHAT CLIENTELE CARON",
            "EMPRUNT", "GENERALI - CT 563802057", "REDEVANCE COLLABORATION", "LOCATION LD AUTRE QUE VÉHICULE", "COMPTE ASSOCIE N°1",
            "FRAIS DE FORMATION SEMINAIRE", "EMPRUNT QASHQAI", "REPAS DE MIDI", "ATECA AUTO", "COTISATIONS COMPLEMENTAIRES RETRAITE",
            "PERTE EMPLOI MADELIN", "VOITURE", "HONORAIRES INFI. 1", "CAISSE", "LOCATION AUTRE QUE VÉHICULE", "PRODUITS ENTRETIEN",
            "CHEQUES VACANCES", "CONSOMABLE F. BUREAU", "COMPLEMENTAIRE FRAIS DE SANTE", "RETRAITE AXA", "MINI", "HONORAIRES Infirmier 1",
            "RACHAT PATIENTELE", "PRET PROFESSIONNEL", "HONORAIRES RETROCEDES A REMPLACANT", "EMPRUNT AUTO", "APPORT SCM INF 1 FONDERFLICK",
            "MACSF COMPLEMENTAIRE", "ACHATS MARCHANDISES", "PETIT MATERIEL REUTILISABLE", "DEPENSES MIXTES", "INVESTISSEMENT", "PRET PROF EQUIPEMENT",
            "RACHAT DU VÉHICULE", "PAPETERIE ENCRE", "BICS", "FRAIS DE CONGRES", "CROSSLAND X", " ACHAT C3", "ACHAT PATIENTELLE", "COMPTE FREYBURGER",
            "COMPTE RUBEL", "COMPTE SOLIGNAC", "ACOMPTE SCOTTO DI CARLO", "FRAIS D'ACTES ET DE CONTENTIEUX", "LEASING AUTRE QUE VÉHICULE", "MADELIN",
            "CHARGES DIVERSES", "I3 BMW", "EMPRUNT BMW", "AUTO RENAULT CAPTUR", "AUTO CLASSE A 200", "RACHAT DE CLIENTELE", "EMPRUNTS CLIENTELE",
            "SOGEFINANCE", "PRET TRAVAUX", "PRET CABINET", "PRET CLIMATISATION CABINET", "PRET AUTO", "FRAIS DE VOITURE (REEL)", "ACHATS MARCHANDISE",
            "RESERVE MEDITRESOR CMV", "C4 PICASSO", "EMPRUNT MAZDA CX3", "EMPRUNT C4AIR CROSS", "LOCAL", "SUZUKI", "VEHICULE", "VOITURE KUGA",
            "PRET MOKKA", "ACHAT VOITURE", "RACHAT CLIENTELE", "RÉINTÉGRATION COMPTABLE", "CAPITAL", "IMMO.MATERIELS DE BUREAU ET INFORMATIQUE",
            "COTISATIONS MADELIN", "COMPTE ASSOCIES", "EMPRUNT ACHAT ZOE", "LOCATION BATTERIE ZOE", "REMBT EMPRUNT ZOE 1ER", "INDMNITES RBT EMPT",
            "ZOE 2 EME", "RACHAT AUTO ET PERSO", "VEHICULE CLIO IV", "ORDINATEUR MAC", "PRÊT AUTOMOBILE", "EMPRUNT OPEL MOKKA", "AGIPI - RETRAITE COMPLEMENTAIRE",
            "VÉHICULE", "CPLMNT TRESORERIE", "COMPTE D'ATTENTE DES IMMOBILISATIONS", "BNP", "JEEP", "RACHAT CLIENTÈLE", "AUTO MINI", "ACHAT FORD", "PATIENTEL",
            "CITROEN DS 3", "PRET PERSONNEL", "PRÊT AUTO PEUGEOT 3008", "AUDI A3", "PRÊT BNPPARIBAS", "EMPRUNT DEBUT ACTIVITE", "ACHAT NOUVEAU VEHICULE",
            "LOCATION DEFIBRILLATEUR", "EMPRUNT PATIENTELE 85% DEDUCTIBLE", "AUTOMOBILE", "EMPRUNT PATIENTEL", "PRET RACHAT PATIENTELE 2016",
            "ACHAT CITROEN C4", "ENVELOPPE 3", "EQUIPEMENT", "ENVELOPPE", "COMPTE ASSOCIE N°4", "RACHAT DE LA PATIENTELE", "REPAS DU MIDI",
            "FINANCEMENT TOYOTA AYGO", "EMPRUNT CLIENTELE 95% DEDUCTIBLE", "ACHAT VEHICULE", "ACHAT VOITURE NISSAN QASHQAI", "EMPRUNT INSTALLATION",
            "DIAC VOITURE", "SUBARU LEVORG", "EMPRUNT TOYOTA YARIS", "PATIENTELE + TRESORERIE 46.66% DEDUCTIBL", "DACIA 2", "DACIA 3", "TRESORERIE",
            "AGIOS", "MOBILIER + OUTILLAGE", "BMW X5", "CABINET", "ACHAT PATIENTELE", "VEHICULE DACIA", "COTISATIONS PROFESSIONNELLES", "PRÊT PRO",
            "HONORAIRES RÉTROCÉDÉS JEREMY", "HONORAIRES RÉTROCÉDÉS SANDRA", "HONORAIRES RÉTROCÉDÉS SYLVIA", "RETRAITE MACSF", "APPORT PERSONNEL",
            "MEDIFORCE", "PRET CIC", "ACHAT VOITURE EOS", "EMPRUNT CLIO 2013", "AIDE A LA TRESORIE", "CITROEN C3", "CREDIT RACHAT DE PATIENTELE",
            "EMPRUNT PRO VOITURE", "PRET PRO", "EMPRUNTNAUTOMOBILE", "PRET VOITURE", "PRET PROFESSIONEL", "ENTRETIEN ET REPARATIONS cab inf.",
            "EMPRUNT POLO", "CLIENTELE", "COMPTE D'ATTENTE", "LOCATION VEHICULE BIC", "LEASING mini", "ALFA", "AUTO", "ACHAT CABINET", "PRÊT VOITURE",
            "ACHAT TIGUAN", "EMPRUNT PRO", "PRET VOITURE PERSO", "CREDIT SUZUKI VITARA", "INSTALLATION"
    };

    private ParagraphVectors paragraphVectors;
    private boolean avecMontant = false;
    private List<String> categoriesLoaded = null;

    public CSVRetouches() {
    }

    public static LocalTransformProcessRecordReader get(String csvFileClasspath) throws IOException, InterruptedException {
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Id")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "Categorie", "Compte")
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
                .addColumnsString("Categorie")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "Compte")
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

        int nbLignesRetirees = 0;
        List<String> lignes = Lists.newArrayList();
        List<String> categoriesRetirees = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            String cat = next.get(2).toString();
            String libelle = next.get(1).toString();
            String compte = next.get(3).toString();

            // Règles de retrait (ex. chèques)
            if (aRetirer(cat, libelle)) {
                nbLignesRetirees++;
                continue;
            }

            Apollon apollon = Apollon.fromNumero(compte);
            if (apollon == null) {
                // 2eme check avec label cette fois
                // On traite les cas de numéro racine connus (ex. 455XXXX, 164XXXX)
                cat = adapteCategorie(cat, libelle, compte);
                apollon = Apollon.fromLabel(cat);

                if (apollon == null) {
                    if (cat != null && !categoriesRetirees.contains(cat)) {
                        categoriesRetirees.add(cat);
                    }
                    nbLignesRetirees++;
                    continue; // On ne connait pas la catégorie on zap la ligne
                }
            }

            String ligne = "\"" + apollon.getLabel() + "\",\"" + libelle + "\",\"" + next.get(0).toString() + "\",\"" + compte + "\"";
            lignes.add(ligne);
        }

        System.out.println("Traitement - retrait de " + nbLignesRetirees + " lignes");
        for (String categoriesRetiree : categoriesRetirees) {
            System.out.println("CATEG retirée : " + categoriesRetiree);
        }

        // Enregistrement
        String outputPath = "apollon_data_2018.normalise.csv";
        FileWriter fw = new FileWriter(outputPath);
        for (String ligne : lignes) {
            fw.write(ligne + "\n");
        }
        fw.close();
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
            String cat = next.get(2).toString();
            String libelle = next.get(1).toString();
            String ligne = "\"" + cat + "\",\"" + libelle + "\",\"" + next.get(0).toString() + "\"";
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

    private static boolean aRetirer(String cat, String libelle) {
        return cat.isEmpty()
                || (libelle.contains("CHEQUE") && !libelle.contains("VIR"));
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
        if ("IMMO.MATERIELS DE BUREAU ET INFORMATIQUE".equals(cat)) return "";
        if ("COTISATIONS MADELIN".equals(cat)) return "";
        if ("COMPTE ASSOCIES".equals(cat)) return "";
        if ("AGIPI - RETRAITE COMPLEMENTAIRE".equals(cat)) return "";
        if ("COMPTE D'ATTENTE DES IMMOBILISATIONS".equals(cat)) return "";
        if ("COMPTE ASSOCIE N°4".equals(cat)) return "";
        if ("REPAS DU MIDI".equals(cat)) return "";
        if ("AGIOS".equals(cat)) return "";
        if ("COTISATIONS PROFESSIONNELLES".equals(cat)) return "";
        if ("HONORAIRES RÉTROCÉDÉS JEREMY".equals(cat)) return "";
        if ("HONORAIRES RÉTROCÉDÉS SANDRA".equals(cat)) return "";
        if ("HONORAIRES RÉTROCÉDÉS SYLVIA".equals(cat)) return "";
        if ("RETRAITE MACSF".equals(cat)) return "";
        if ("ENTRETIEN ET REPARATIONS cab inf.".equals(cat)) return "";
        if ("COMPTE D'ATTENTE".equals(cat)) return "";
        if ("LEASING mini".equals(cat)) return "";



        return cat;
    }
}
