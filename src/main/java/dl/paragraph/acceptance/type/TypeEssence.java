package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Resultat;

public class TypeEssence implements Acceptance {

    public static final String CATEG_1 = Apollon.CARBURANT_VEHICULE.getLabel();

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
