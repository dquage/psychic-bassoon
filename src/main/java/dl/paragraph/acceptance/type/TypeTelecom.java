package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Resultat;

public class TypeTelecom implements Acceptance {

    public static final String CATEG_1 = Apollon.FRAIS_DE_TELECOMMUNICATION.getLabel();

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
