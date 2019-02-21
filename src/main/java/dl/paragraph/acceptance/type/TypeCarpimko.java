package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Resultat;

public class TypeCarpimko implements Acceptance {

    public static final String CATEG_1 = "COTISATION_CARPIMKO";

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
