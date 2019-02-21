package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Resultat;

public class TypeEssence implements Acceptance {

    public static final String CATEG_1 = "ESSENCE_VEHICULE";

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
