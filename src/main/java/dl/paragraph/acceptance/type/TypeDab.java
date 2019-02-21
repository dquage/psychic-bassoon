package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Resultat;

public class TypeDab implements Acceptance {

    public static final String CATEG_1 = "COMPTE_DE_L_EXPLOITANT";

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
