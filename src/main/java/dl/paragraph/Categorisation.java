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
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Categorisation {

    public static String[] CATEGORIES = {
            "SERVICES_BANCAIRES", "COMPTE_DE_L_EXPLOITANT", "DEPLACEMENTS_MISSIONS_RECEPTIONS", "REDEVANCE_COLLABORATION", "ESSENCE_VEHICULE",
            "CHARGES_DIVERSES_DE_GESTION_COURANTE", "REPAS_DE_MIDI", "ENTRETIEN_ET_REPARATIONS", "DEPENSES_VEHICULE",
            "EMPRUNT", "COTISATIONS_COMPLEMENTAIRES_EXPLOITANT", "CHARGES_FINANCIERES", "ASSURANCE_VEHICULE", "FRAIS_DE_TELECOMMUNICATION",
            "HONORAIRES_RETROCEDES", "FOURNITURES_DE_BUREAU_ET_ADMINISTRATIVES", "PETIT_OUTILLAGE_PETIT_MATERIEL", "ASSURANCES",
            "COTISATIONS_PROF_ET_SYNDICALE", "FRAIS_POSTAUX", "CFE_CVAE", "ACHATS_A_USAGE_UNIQUE", "TRANSPORTS_DIVERS__BUS_TAXIS_",
            "HONORAIRES_RETROCEDES_A_REMPLACANT", "CADEAUX_CLIENTELE", "COTISATION_CARPIMKO", "FRAIS_DE_FORMATION_SEMINAIRE",
            "CHEQUES_VACANCES", "FRAIS_DE_DOCUMENTATION_TECHNIQUE", "LOCATIONS_IMMOBILIERES", "LLD_VEHICULE", "AUTRES_SERVICES_EXTERIEURS_HONORAIRES",
            "HONORAIRES", "AGIOS_BANCAIRES", "PRESTATIONS_DE_SERVICE", "COTISATIONS_MALADIE_EXPLOITANT",
            "COMPTE_ASSOCIE", "AUTRES_IMPOTS", "LEASING_AUTRE_QUE_VEHICULE", "EDF_GDF_CHAUFFAGE", "REMUNERATIONS_DU_PERSONNEL",
            "COTISATIONS_URSSAF_DE_L_EXPLOITANT", "PRODUITS_D_ENTRETIEN", "LEASING_VEHICULE", "COTISATIONS_COMPLEMENTAIRES_SANTE", "PART_PRIVEE_DES_DEPENSES_MIXTES",
            "PERTE_EMPLOI_MADELIN", "FRAIS_DE_CONGRES", "LOCATION_LD_AUTRE_QUE_VEHICULE", "COTISATIONS_COMPLEMENTAIRES_RETRAITE",
            "FRAIS_D_ACTES_ET_DE_CONTENTIEUX", "FOURNISSEURS_D_IMMOBILISATIONS", "ACHATS_MARCHANDISES", "COMPTE_D_ATTENTE", "APPORT_PERSONNEL",
            "SECURITE_SOC_ET_PREVOYANCE_DU_PERSONNEL", "COTISATIONON_CARPIMKO", "CREDIT_BAIL_LEASING_MOBILIER"
    };

    private ParagraphVectors paragraphVectors;
    private boolean avecMontant = false;

    public Categorisation() {
    }


    public void train() throws IOException, InterruptedException {

        LabelAwareIterator iterator = readCSVDataset("depenses2017.data", avecMontant);

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

    public void evaluate(List<Record> records) throws IOException, InterruptedException {

        if (paragraphVectors == null) {
            paragraphVectors = readModelFromFile();
        }

        LabelAwareIterator iterator = readDataForEvaluation(records);

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);

        LabelSeeker seeker = new LabelSeeker(
                Arrays.asList(CATEGORIES),
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

            Pair<String, Double> score1 = scores.get(0);
            Pair<String, Double> score2 = scores.get(1);
            Pair<String, Double> score3 = scores.get(2);

            Resultat resultat = Resultat.builder()
                    .libelle(document.getContent())
                    .record(record)
                    .categorie1(Categorie.builder().label(score1.getFirst()).score(score1.getSecond()).build())
                    .categorie2(Categorie.builder().label(score2.getFirst()).score(score2.getSecond()).build())
                    .categorie3(Categorie.builder().label(score3.getFirst()).score(score3.getSecond()).build())
                    .build();

            resultat.setLabelAccepted(acceptanceType.accept(resultat));

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
    public static LabelAwareIterator readCSVDataset(String csvFileClasspath, boolean avecMontant) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
//                .addColumnsString("id", "numcba", "montant", "libele", "numcompte", "categorie")
                //Or for convenience define multiple columns of the same type
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "LibelleSimple", "Categorie")
                .build();

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Id", "NumCba", "NumCompte")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        LabelsSource labelsSource = new LabelsSource(getCategories(transformProcessRecordReader)); // On passe la liste des catégories de chaque libellé ordonnée
        MySentenceIterator iterator = new MySentenceIterator(transformProcessRecordReader, avecMontant);
        LabelAwareIterator myDataSetIterator = new SentenceIteratorConverter(iterator, labelsSource);
//        myDataSetIterator.setCollectMetaData(true);
        return myDataSetIterator;
    }

    private static List<String> getCategories(LocalTransformProcessRecordReader transformProcessRecordReader) {
        List<String> categories = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            categories.add(transformProcessRecordReader.next().get(2).toString());
        }
        transformProcessRecordReader.reset();
        return categories;
    }

    public static List<Record> readCSVRecordsTest(String csvFileClasspath) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "LibelleSimple", "Categorie")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Id", "NumCba", "LibelleSimple")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        List<Record> records = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            records.add(Record.builder().categorie(next.get(2).toString()).libelle(next.get(1).toString()).montant(next.get(0).toDouble()).build());
        }
        return records;
    }

    public static List<Record> readCSVRecordsReels(String csvFileClasspath) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "LibelleSimple")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Id", "NumCba")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        List<Record> records = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            List<Writable> next = transformProcessRecordReader.next();
            records.add(Record.builder().libelle(next.get(1).toString()).montant(next.get(0).toDouble()).build());
        }
        return records;
    }

    private static LabelAwareIterator readDataForEvaluation(List<Record> records) {

        MySentenceFromListIterator iterator = new MySentenceFromListIterator(records);
        return new MySentenceIteratorConverter(iterator);
    }
}
