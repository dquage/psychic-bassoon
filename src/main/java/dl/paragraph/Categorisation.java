package dl.paragraph;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
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

    ParagraphVectors paragraphVectors;

    public Categorisation() {
    }


    public void train() throws IOException, InterruptedException {

        LabelAwareIterator iterator = readCSVDataset("depenses2017.data");

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(tokenizerFactory)
                .build();

        // Start model training
        System.out.println("Train starting...");
        Stopwatch started = Stopwatch.createStarted();
        paragraphVectors.fit();
        System.out.println("Train done In " + started.stop().elapsed(TimeUnit.SECONDS) + "s");

        saveModel(paragraphVectors);
    }

    public void evaluate() throws IOException, InterruptedException {

        if (paragraphVectors == null) {
            paragraphVectors = readModelFromFile();
        }

        LabelAwareIterator iterator = readCSVDataset("depenses2017.test");

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);

        LabelSeeker seeker = new LabelSeeker(
                Arrays.asList(CATEGORIES),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        System.out.println("Evaluation starting...");
//        int limit = 10;
//        while (iterator.hasNextDocument() && limit > 0) {
//            LabelledDocument document = iterator.nextDocument();
//            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
//            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
//            System.out.println("Item [" + document.getContent() + "] falls into the following categories: ");
//            for (int i = 0; i < 3 && i < scores.size(); i++) {
//                Pair<String, Double> score = scores.get(i);
//                System.out.println("        " + score.getFirst() + ": " + score.getSecond());
//            }
//            limit--;
//        }

        List<Resultat> resultats = Lists.newArrayList();
        while (iterator.hasNextDocument()) {

            LabelledDocument document = iterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            Pair<String, Double> score1 = scores.get(0);
            Pair<String, Double> score2 = scores.get(1);
            Pair<String, Double> score3 = scores.get(2);
            resultats.add(Resultat.builder()
                    .libelle(document.getContent())
                    .categorie1(Categorie.builder().label(score1.getFirst()).score(score1.getSecond()).build())
                    .categorie2(Categorie.builder().label(score2.getFirst()).score(score2.getSecond()).build())
                    .categorie3(Categorie.builder().label(score3.getFirst()).score(score3.getSecond()).build())
                    .build());
        }

        saveResultats(resultats);
        saveResultatsParCategorie(resultats);
    }

    private void saveResultats(List<Resultat> resultats) throws IOException {

        StringBuilder sb = new StringBuilder();
        for (Resultat resultat : resultats) {
            sb.append(resultat.getLibelle());
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

    private void saveResultatsParCategorie(List<Resultat> resultats) throws IOException {

        StringBuilder sb = new StringBuilder();
        for (String category : CATEGORIES) {
            sb.append(category);
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
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static LabelAwareIterator readCSVDataset(String csvFileClasspath) throws IOException, InterruptedException {

        Schema inputDataSchema = new Schema.Builder()
//                .addColumnsString("id", "numcba", "montant", "libele", "numcompte", "categorie")
                //Or for convenience define multiple columns of the same type
                .addColumnsString("Id", "NumCba")
                .addColumnDouble("Montant")
                .addColumnsString("Libelle", "NumCompte")
                .addColumnCategorical("Categorie", Arrays.asList(CATEGORIES))
                .build();

        // Permet de conditionner, transformer, enlever les data que l'on récupère du CSV
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("Id", "NumCba", "Montant", "NumCompte")
                .build();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        LocalTransformProcessRecordReader transformProcessRecordReader = new LocalTransformProcessRecordReader(rr, tp);

        LabelsSource labelsSource = new LabelsSource(getCategories(transformProcessRecordReader)); // On passe la liste des catégories de chaque libellé ordonnée
        MySentenceIterator iterator = new MySentenceIterator(transformProcessRecordReader);
        LabelAwareIterator myDataSetIterator = new SentenceIteratorConverter(iterator, labelsSource);

//        myDataSetIterator.setCollectMetaData(true);
        return myDataSetIterator;
    }

    private static List<String> getCategories(LocalTransformProcessRecordReader transformProcessRecordReader) {
        List<String> categories = Lists.newArrayList();
        while (transformProcessRecordReader.hasNext()) {
            categories.add(transformProcessRecordReader.next().get(1).toString());
        }
        transformProcessRecordReader.reset();
        return categories;
    }
}
