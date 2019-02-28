package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Record;
import dl.paragraph.pojo.Resultat;

public class TypeCba implements Acceptance {

    public static final String CATEG_1 = Apollon.ENTRETIEN_ET_REPARATIONS.getLabel();
    public static final String CATEG_2 = Apollon.LOCATION_LD_AUTRE_QUE_VEHICULE.getLabel();

    @Override
    public String accept(Resultat resultat) {
        Record record = resultat.getRecord();
        if (record != null && record.isDepense()) {
            return CATEG_1;
        }
        return null;
    }
}
