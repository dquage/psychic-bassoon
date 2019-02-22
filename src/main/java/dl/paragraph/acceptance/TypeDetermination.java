package dl.paragraph.acceptance;

public class TypeDetermination {

    public static final String CBA = "CBA";
    public static final String ESSENCE = "ESSENCE";
    public static final String ASSURANCE = "ASSURANCE";
    public static final String TELECOM = "TELECOM";
    public static final String CARPIMKO = "CARPIMKO";
    public static final String TRANSPORTS = "TRANSPORTS";
    public static final String DAB = "DAB";
    public static final String CPAM = "CPAM";
    public static final String MUTUELLE = "MUTUELLE";

    public static String determineType(String libelle) {

        String libnorm = libelle.trim().toLowerCase();
        if (isCba(libnorm)) {
            return CBA;
        } else if (isEssence(libnorm)) {
            return ESSENCE;
        } else if (isAssurance(libnorm)) {
            return ASSURANCE;
        } else if (isTelecommunication(libnorm)) {
            return TELECOM;
        } else if (isCarpimko(libnorm)) {
            return CARPIMKO;
        } else if (isTransports(libnorm)) {
            return TRANSPORTS;
        } else if (isDab(libnorm)) {
            return DAB;
        } else if (isCpam(libnorm)) {
            return CPAM;
        } else if (isMutuelle(libnorm)) {
            return MUTUELLE;
        }
        return null;
    }

    public static boolean isCba(String libelle) {
        return libelle.contains("cba informatique");
    }

    public static boolean isEssence(String libelle) {
        return libelle.contains("essence") || libelle.contains("dac") || libelle.contains("carb");
    }

    public static boolean isAssurance(String libelle) {
        return libelle.contains("assurance");
    }

    public static boolean isTelecommunication(String libelle) {
        return libelle.contains("free") || libelle.contains("bouygues") || libelle.contains("adsl") || libelle.contains("orange") || libelle.contains("sfr");
    }

    public static boolean isCarpimko(String libelle) {
        return libelle.contains("c a r p i m") || libelle.contains("carpimko");
    }

    public static boolean isTransports(String libelle) {
        return libelle.contains("autoroute") || libelle.contains("taxi") || libelle.contains("uber") || libelle.contains("sncf") || libelle.contains("park");
    }

    private static boolean isDab(String libelle) {
        return libelle.contains("dab") || libelle.contains("retrait");
    }

    private static boolean isCpam(String libelle) {
        return libelle.contains("cpam");
    }

    private static boolean isMutuelle(String libelle) {
        return libelle.contains("mutuelle");
    }
}
