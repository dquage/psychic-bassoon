package dl.paragraph.acceptance;

import dl.paragraph.acceptance.type.*;
import dl.paragraph.pojo.Resultat;

public class AcceptanceType implements Acceptance {

    private Acceptance base = new AcceptanceComparative();

    public AcceptanceType(Acceptance base) {
        this.base = base;
    }

    @Override
    public String accept(Resultat resultat) {

        String categorie = base.accept(resultat);
        if (categorie != null) {
            return categorie;
        }

        String type = TypeDetermination.determineType(resultat.getLibelle());
        if (type == null) {
            return null;
        }

        switch (type) {
            case TypeDetermination.CBA:
                return new TypeCba().accept(resultat);
            case TypeDetermination.ESSENCE:
                return new TypeEssence().accept(resultat);
            case TypeDetermination.CARPIMKO:
                return new TypeCarpimko().accept(resultat);
            case TypeDetermination.TELECOM:
                return new TypeTelecom().accept(resultat);
            case TypeDetermination.TRANSPORTS:
                return new TypeTransports().accept(resultat);
            case TypeDetermination.ASSURANCE:
                return new TypeAssurance().accept(resultat);
            case TypeDetermination.DAB:
                return new TypeDab().accept(resultat);
            case TypeDetermination.CPAM:
                return new TypeHonoraires().accept(resultat);
            case TypeDetermination.MUTUELLE:
                return new TypeHonoraires().accept(resultat);
        }
        return null;
    }
}
