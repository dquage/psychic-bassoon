package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Resultat;

public class TypeTelecom implements Acceptance {

    public static final String CATEG_1 = "FRAIS_DE_TELECOMMUNICATION";

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
