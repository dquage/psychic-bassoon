package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Record;
import dl.paragraph.pojo.Resultat;

public class TypeHonoraires implements Acceptance {

    public static final String CATEG_1 = Apollon.HONORAIRES.getLabel();

    @Override
    public String accept(Resultat resultat) {

        Record record = resultat.getRecord();
        if (record != null && record.isRecette()) {
            return CATEG_1;
        }
        return null;
    }
}
