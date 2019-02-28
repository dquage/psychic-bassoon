package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import dl.paragraph.acceptance.AcceptanceComparative;
import dl.paragraph.acceptance.AcceptanceType;
import dl.paragraph.my.MySentenceFromListIterator;
import dl.paragraph.my.MySentenceIterator;
import dl.paragraph.my.MySentenceIteratorConverter;
import dl.paragraph.pojo.Categorie;
import dl.paragraph.pojo.Record;
import dl.paragraph.pojo.Resultat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class Categorisation {

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

    public Categorisation() {
    }


    public void train() throws IOException, InterruptedException {

        LabelAwareIterator iterator = readCSVDataset("apollon_data_2018.train.csv", avecMontant);

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        paragraphVectors = ConfigsTests.config2c(iterator, tokenizerFactory);

        // Start model training
        System.out.println("Train starting...");
        Stopwatch started = Stopwatch.createStarted();
        paragraphVectors.fit();
        System.out.println("Train done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");

        saveModel(paragraphVectors);
    }

    public void evaluate() throws Exception {

        evaluate(readCSVRecordsTest("apollon_data_2018.test.csv"));
    }

    public void evaluate(List<Record> records) throws Exception {

        if (paragraphVectors == null) {
            paragraphVectors = readModelFromFile();
        }

        if (categoriesLoaded == null) {
            throw new Exception("Il faut d'abord renseigner la liste complète des catégories pour évaluer");
        }

        LabelAwareIterator iterator = readDataForEvaluation(records);

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);

        LabelSeeker seeker = new LabelSeeker(
                categoriesLoaded,
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        Stopwatch started = Stopwatch.createStarted();
        System.out.println("Evaluation starting...");

//        int limit = 10;
//        while (iterator.hasNextDocument() && limit > 0) {
//            LabelledDocument document = iterator.nextDocument();
//            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
//            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
//            System.out.println("Item [" + document.getContent() + "] falls into the following categories: ");
//            for (int i = 0; i < scores.size(); i++) {
//                Pair<String, Double> score = scores.get(i);
//                System.out.println("        " + score.getFirst() + ": " + score.getSecond());
//            }
//            limit--;
//        }
//        iterator.reset();

        AcceptanceType acceptanceType = new AcceptanceType(new AcceptanceComparative());

        List<Resultat> resultats = Lists.newArrayList();
        while (iterator.hasNextDocument()) {

            LabelledDocument document = iterator.nextDocument();
            Record record = ((MySentenceIteratorConverter) iterator).record();

            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            Resultat resultat = null;
            if (scores != null) {
                Pair<String, Double> score1 = scores.get(0);
                Pair<String, Double> score2 = scores.get(1);
                Pair<String, Double> score3 = scores.get(2);
                resultat = Resultat.builder()
                        .libelle(document.getContent())
                        .record(record)
                        .categorie1(Categorie.builder().label(score1.getFirst()).score(score1.getSecond()).build())
                        .categorie2(Categorie.builder().label(score2.getFirst()).score(score2.getSecond()).build())
                        .categorie3(Categorie.builder().label(score3.getFirst()).score(score3.getSecond()).build())
                        .build();
                resultat.setLabelAccepted(acceptanceType.accept(resultat));
            } else {
                resultat = Resultat.builder()
                        .libelle(document.getContent())
                        .record(record)
                        .categorie1(null)
                        .categorie2(null)
                        .categorie3(null)
                        .build();
            }

            resultats.add(resultat);
        }
        System.out.println("Evaluation done in " + started.stop().elapsed(TimeUnit.SECONDS) + "s");

        Tracing.saveResultats(resultats);
        Tracing.saveResultatsParCategorie(resultats);
    }

    public void saveModel(ParagraphVectors model) {
        String path = "./model/paragraph_model.bin";
        WordVectorSerializer.writeParagraphVectors(model, path);
    }

    private ParagraphVectors readModelFromFile() throws IOException {
        File file = new File("model/paragraph_model.bin");
        if (file.exists()) {
            return WordVectorSerializer.readParagraphVectors(file);
        }
        return null;
    }

    /**
     * Lecture d'un fichier CSV formatté (si formatage change, cette fonction doit changer).
     * Transformation des données du csv à la lecture pour une meilleure interprétation.
     * Création d'un iterator de dataSet sur les données.
     * Iterator perso qui gère les données texte et se charge de créer les dataSet, voir MyDataSetIterator.
     *
     * @param csvFileClasspath
     * @param avecMontant Indique si on ajoute le montant au libellé
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public LabelAwareIterator readCSVDataset(String csvFileClasspath, boolean avecMontant) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
//                .addColumnsString("id", "numcba", "montant", "libele", "numcompte", "categorie")
                //Or for convenience define multiple columns of the same type
                .addColumnsString("Categorie", "Libelle")
                .addColumnDouble("Montant")
                .addColumnsString("Compte")
                .build();

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .build();


        // On doit avoir les 3 informations suivantes dans cet ordre :
        // - Montant
        // - Libellé
        // - Catégorie
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        categoriesLoaded = getCategories(transformProcessRecordReader);
        LabelsSource labelsSource = new LabelsSource(categoriesLoaded); // On passe la liste des catégories de chaque libellé ordonnée
        MySentenceIterator iterator = new MySentenceIterator(transformProcessRecordReader, avecMontant);
        LabelAwareIterator myDataSetIterator = new SentenceIteratorConverter(iterator, labelsSource);
//        myDataSetIterator.setCollectMetaData(true);
        return myDataSetIterator;
    }

    private static List<String> getCategories(LocalTransformProcessRecordReader transformProcessRecordReader) {
        List<String> categories = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            String cat = transformProcessRecordReader.next().get(2).toString();
            if (!categories.contains(cat)) {
                categories.add(cat);
            }
        }
        transformProcessRecordReader.reset();
        return categories;
    }

    public List<Record> readCSVRecordsTest(String csvFileClasspath) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Categorie", "Libelle")
                .addColumnDouble("Montant")
                .addColumnsString("Compte")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        categoriesLoaded = getCategories(transformProcessRecordReader);

        List<Record> records = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            records.add(Record.builder().categorie(next.get(0).toString()).libelle(next.get(1).toString()).montant(next.get(2).toDouble()).build());
        }
        return records;
    }

    private static LabelAwareIterator readDataForEvaluation(List<Record> records) {

        MySentenceFromListIterator iterator = new MySentenceFromListIterator(records);
        return new MySentenceIteratorConverter(iterator);
    }
}
